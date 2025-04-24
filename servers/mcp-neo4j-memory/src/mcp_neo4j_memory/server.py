import os
import logging
import json
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

import neo4j
from neo4j import GraphDatabase
from pydantic import BaseModel

import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio

# Set up logging
logger = logging.getLogger('mcp_neo4j_memory')
logger.setLevel(logging.INFO)

# Models for our knowledge graph
class Entity(BaseModel):
    name: str
    type: str
    observations: List[str]

class Relation(BaseModel):
    source: str
    target: str
    relationType: str

class KnowledgeGraph(BaseModel):
    entities: List[Entity]
    relations: List[Relation]

class ObservationAddition(BaseModel):
    entityName: str
    contents: List[str]

class ObservationDeletion(BaseModel):
    entityName: str
    observations: List[str]

class Neo4jMemory:
    def __init__(self, neo4j_driver):
        self.neo4j_driver = neo4j_driver
        self.create_fulltext_index()

    def create_fulltext_index(self):
        try:
            # TODO , 
            query = """
            CREATE FULLTEXT INDEX search IF NOT EXISTS FOR (m:Memory) ON EACH [m.name, m.type, m.observations];
            """
            self.neo4j_driver.execute_query(query)
            logger.info("Created fulltext search index")
        except neo4j.exceptions.ClientError as e:
            if "An index with this name already exists" in str(e):
                logger.info("Fulltext search index already exists")
            else:
                raise e

    async def load_graph(self, filter_query="*"):
        query = """
            CALL db.index.fulltext.queryNodes('search', $filter) yield node as entity, score
            OPTIONAL MATCH (entity)-[r]-(other)
            RETURN collect(distinct {
                name: entity.name, 
                type: entity.type, 
                observations: entity.observations
            }) as nodes,
            collect(distinct {
                source: startNode(r).name, 
                target: endNode(r).name, 
                relationType: type(r)
            }) as relations
        """
        
        result = self.neo4j_driver.execute_query(query, {"filter": filter_query})
        
        if not result.records:
            return KnowledgeGraph(entities=[], relations=[])
        
        record = result.records[0]
        nodes = record.get('nodes')
        rels = record.get('relations')
        
        entities = [
            Entity(
                name=node.get('name'),
                type=node.get('type'),
                observations=node.get('observations', [])
            )
            for node in nodes if node.get('name')
        ]
        
        relations = [
            Relation(
                source=rel.get('source'),
                target=rel.get('target'),
                relationType=rel.get('relationType')
            )
            for rel in rels if rel.get('source') and rel.get('target') and rel.get('relationType')
        ]
        
        logger.debug(f"Loaded entities: {entities}")
        logger.debug(f"Loaded relations: {relations}")
        
        return KnowledgeGraph(entities=entities, relations=relations)

    async def create_entities(self, entities: List[Entity]) -> List[Entity]:
        # Group entities by type to avoid Cypher syntax error (similar to create_relations)
        entities_by_type = {}
        for entity in entities:
            if entity.type not in entities_by_type:
                entities_by_type[entity.type] = []
            entities_by_type[entity.type].append(entity.model_dump())
        
        # Process entities by type
        for entity_type, entities_of_type in entities_by_type.items():
            # Fix the syntax by using f-string and backticks for the dynamic entity type
            query = f"""
            UNWIND $entities as entity
            MERGE (e:Memory {{ name: entity.name }})
            SET e += entity {{.type, .observations}}
            SET e:`{entity_type}`
            """
            
            self.neo4j_driver.execute_query(query, {"entities": entities_of_type})
        
        return entities

    async def create_relations(self, relations: List[Relation]) -> List[Relation]:
        """
        Create relationships between entities using APOC for dynamic relationship types.
        Requires APOC plugin to be installed.
        """
        logger.info("Using APOC's apoc.create.relationship for dynamic relation creation.")
        
        # Process each relation individually using APOC
        for relation in relations:
            query = """
            MATCH (from:Memory {name: $source}), (to:Memory {name: $target})
            CALL apoc.create.relationship(from, $rel_type, {}, to) YIELD rel
            RETURN type(rel) as created_relation_type
            """
            
            params = {
                "source": relation.source,
                "target": relation.target,
                "rel_type": relation.relationType  # Pass the type as a parameter
            }
            
            try:
                # Execute the query using APOC
                result = self.neo4j_driver.execute_query(query, params)
                # Optional: Log the result or check if the relation was created
                if result.records:
                    created_type = result.records[0].get("created_relation_type")
                    logger.debug(f"Created relation '{relation.source}' -[{created_type}]-> '{relation.target}'")
                else:
                    logger.warning(f"Could not create relation: {relation.source} -[{relation.relationType}]-> {relation.target}. Nodes might not exist.")
            except neo4j.exceptions.ClientError as e:
                # Handle potential errors, e.g., APOC procedure not found
                if "There is no procedure with the name `apoc.create.relationship`" in str(e):
                    logger.error("APOC procedure 'apoc.create.relationship' not found. Ensure APOC plugin is installed and the database restarted.")
                    # Potentially re-raise or handle more gracefully
                    raise e 
                else:
                    logger.error(f"Failed to create relation {relation.source} -> {relation.target}: {e}")
                    # Re-raise other client errors
                    raise e
            except Exception as e:
                logger.error(f"An unexpected error occurred during relation creation: {e}")
                # Re-raise unexpected errors
                raise e
        
        return relations

    async def add_observations(self, observations: List[ObservationAddition]) -> List[Dict[str, Any]]:
        query = """
        UNWIND $observations as obs  
        MATCH (e:Memory { name: obs.entityName })
        WITH e, [o in obs.contents WHERE NOT o IN e.observations] as new
        SET e.observations = coalesce(e.observations,[]) + new
        RETURN e.name as name, new
        """
            
        result = self.neo4j_driver.execute_query(
            query, 
            {"observations": [obs.model_dump() for obs in observations]}
        )

        results = [{"entityName": record.get("name"), "addedObservations": record.get("new")} for record in result.records]
        return results

    async def delete_entities(self, entity_names: List[str]) -> None:
        query = """
        UNWIND $entities as name
        MATCH (e:Memory { name: name })
        DETACH DELETE e
        """
        
        self.neo4j_driver.execute_query(query, {"entities": entity_names})

    async def delete_observations(self, deletions: List[ObservationDeletion]) -> None:
        query = """
        UNWIND $deletions as d  
        MATCH (e:Memory { name: d.entityName })
        SET e.observations = [o in coalesce(e.observations,[]) WHERE NOT o IN d.observations]
        """
        self.neo4j_driver.execute_query(
            query, 
            {
                "deletions": [deletion.model_dump() for deletion in deletions]
            }
        )

    async def delete_relations(self, relations: List[Relation]) -> None:
        # Group relations by relationType to avoid Cypher syntax error
        relations_by_type = {}
        for relation in relations:
            if relation.relationType not in relations_by_type:
                relations_by_type[relation.relationType] = []
            relations_by_type[relation.relationType].append(relation.model_dump())
        
        # Execute query for each relation type
        for rel_type, rels in relations_by_type.items():
            query = f"""
            UNWIND $relations as relation
            MATCH (source:Memory)-[r:`{rel_type}`]->(target:Memory)
            WHERE source.name = relation.source
            AND target.name = relation.target
            DELETE r
            """
            
            self.neo4j_driver.execute_query(query, {"relations": rels})

    async def read_graph(self) -> KnowledgeGraph:
        logger.info("UPDATED CODE: Running read_graph with our fixed version (no arguments needed)")
        return await self.load_graph()

    async def search_nodes(self, query: str) -> KnowledgeGraph:
        return await self.load_graph(query)

    async def find_nodes(self, names: List[str]) -> KnowledgeGraph:
        return await self.load_graph("name: (" + " ".join(names) + ")")

async def main(neo4j_uri: str, neo4j_user: str, neo4j_password: str):
    logger.info("==== RUNNING UPDATED SERVER.PY WITH FIXED ARGUMENT HANDLING ====")
    logger.info(f"Connecting to neo4j MCP Server with DB URL: {neo4j_uri}")

    # Connect to Neo4j
    neo4j_driver = GraphDatabase.driver(
        neo4j_uri,
        auth=(neo4j_user, neo4j_password)
    )
    
    # Verify connection
    try:
        neo4j_driver.verify_connectivity()
        logger.info(f"Connected to Neo4j at {neo4j_uri}")
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}")
        exit(1)

    # Initialize memory
    memory = Neo4jMemory(neo4j_driver)
    
    # Create MCP server
    server = Server("mcp-neo4j-memory")

    # Register handlers
    @server.list_tools()
    async def handle_list_tools() -> List[types.Tool]:
        return [
            types.Tool(
                name="create_entities",
                description="Create multiple new entities in the knowledge graph",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "entities": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string", "description": "The name of the entity"},
                                    "type": {"type": "string", "description": "The type of the entity"},
                                    "observations": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "An array of observation contents associated with the entity"
                                    },
                                },
                                "required": ["name", "type", "observations"],
                            },
                        },
                    },
                    "required": ["entities"],
                },
            ),
            types.Tool(
                name="create_relations",
                description="Create multiple new relations between entities in the knowledge graph. Relations should be in active voice",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "relations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "source": {"type": "string", "description": "The name of the entity where the relation starts"},
                                    "target": {"type": "string", "description": "The name of the entity where the relation ends"},
                                    "relationType": {"type": "string", "description": "The type of the relation"},
                                },
                                "required": ["source", "target", "relationType"],
                            },
                        },
                    },
                    "required": ["relations"],
                },
            ),
            types.Tool(
                name="add_observations",
                description="Add new observations to existing entities in the knowledge graph",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "observations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "entityName": {"type": "string", "description": "The name of the entity to add the observations to"},
                                    "contents": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "An array of observation contents to add"
                                    },
                                },
                                "required": ["entityName", "contents"],
                            },
                        },
                    },
                    "required": ["observations"],
                },
            ),
            types.Tool(
                name="delete_entities",
                description="Delete multiple entities and their associated relations from the knowledge graph",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "entityNames": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "An array of entity names to delete"
                        },
                    },
                    "required": ["entityNames"],
                },
            ),
            types.Tool(
                name="delete_observations",
                description="Delete specific observations from entities in the knowledge graph",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "deletions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "entityName": {"type": "string", "description": "The name of the entity containing the observations"},
                                    "observations": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "An array of observations to delete"
                                    },
                                },
                                "required": ["entityName", "observations"],
                            },
                        },
                    },
                    "required": ["deletions"],
                },
            ),
            types.Tool(
                name="delete_relations",
                description="Delete multiple relations from the knowledge graph",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "relations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "source": {"type": "string", "description": "The name of the entity where the relation starts"},
                                    "target": {"type": "string", "description": "The name of the entity where the relation ends"},
                                    "relationType": {"type": "string", "description": "The type of the relation"},
                                },
                                "required": ["source", "target", "relationType"],
                            },
                            "description": "An array of relations to delete"
                        },
                    },
                    "required": ["relations"],
                },
            ),
            types.Tool(
                name="read_graph",
                description="Read the entire knowledge graph",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            types.Tool(
                name="search_nodes",
                description="Search for nodes in the knowledge graph based on a query",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query to match against entity names, types, and observation content"},
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="find_nodes",
                description="Open specific nodes in the knowledge graph by their names",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "An array of entity names to retrieve",
                        },
                    },
                    "required": ["names"],
                },
            ),
        ]

    @server.call_tool()
    async def handle_call_tool(
        name: str, arguments: Dict[str, Any] | None
    ) -> List[types.TextContent | types.ImageContent]:
        try:
            if name == "create_entities":
                if not arguments or "entities" not in arguments:
                    raise ValueError(f"Missing 'entities' argument for tool: {name}")
                entities = [Entity(**entity) for entity in arguments.get("entities", [])]
                result = await memory.create_entities(entities)
                return [types.TextContent(type="text", text=json.dumps([e.model_dump() for e in result], indent=2))]
                
            elif name == "create_relations":
                if not arguments or "relations" not in arguments:
                    raise ValueError(f"Missing 'relations' argument for tool: {name}")
                relations = [Relation(**relation) for relation in arguments.get("relations", [])]
                result = await memory.create_relations(relations)
                return [types.TextContent(type="text", text=json.dumps([r.model_dump() for r in result], indent=2))]
                
            elif name == "add_observations":
                if not arguments or "observations" not in arguments:
                    raise ValueError(f"Missing 'observations' argument for tool: {name}")
                observations = [ObservationAddition(**obs) for obs in arguments.get("observations", [])]
                result = await memory.add_observations(observations)
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
                
            elif name == "delete_entities":
                if not arguments or "entityNames" not in arguments:
                    raise ValueError(f"Missing 'entityNames' argument for tool: {name}")
                await memory.delete_entities(arguments.get("entityNames", []))
                return [types.TextContent(type="text", text="Entities deleted successfully")]
                
            elif name == "delete_observations":
                if not arguments or "deletions" not in arguments:
                    raise ValueError(f"Missing 'deletions' argument for tool: {name}")
                deletions = [ObservationDeletion(**deletion) for deletion in arguments.get("deletions", [])]
                await memory.delete_observations(deletions)
                return [types.TextContent(type="text", text="Observations deleted successfully")]
                
            elif name == "delete_relations":
                if not arguments or "relations" not in arguments:
                    raise ValueError(f"Missing 'relations' argument for tool: {name}")
                relations = [Relation(**relation) for relation in arguments.get("relations", [])]
                await memory.delete_relations(relations)
                return [types.TextContent(type="text", text="Relations deleted successfully")]
                
            elif name == "read_graph":
                result = await memory.read_graph()
                return [types.TextContent(type="text", text=json.dumps(result.model_dump(), indent=2))]
                
            elif name == "search_nodes":
                if not arguments or "query" not in arguments:
                    raise ValueError(f"Missing 'query' argument for tool: {name}")
                result = await memory.search_nodes(arguments.get("query", ""))
                return [types.TextContent(type="text", text=json.dumps(result.model_dump(), indent=2))]
                
            elif name == "find_nodes":
                if not arguments or "names" not in arguments:
                    raise ValueError(f"Missing 'names' argument for tool: {name}")
                result = await memory.find_nodes(arguments.get("names", []))
                return [types.TextContent(type="text", text=json.dumps(result.model_dump(), indent=2))]
                
            else:
                raise ValueError(f"Unknown tool: {name}")
                
        except Exception as e:
            logger.error(f"Error handling tool call: {e}")
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]

    # Start the server
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        logger.info("MCP Knowledge Graph Memory using Neo4j running on stdio")
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="mcp-neo4j-memory",
                server_version="1.1",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )
