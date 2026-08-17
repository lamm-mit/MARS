from .graph_tools import *
from .utils import *
from .graph_analysis import *

import copy

import re
from IPython.display import display, Markdown

import markdown2
import pdfkit

 
import uuid
import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
import networkx as nx
import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader, PyPDFium2Loader
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import random
from pyvis.network import Network

from tqdm.notebook import tqdm

import itertools
import seaborn as sns
palette = "hls"

import uuid
import pandas as pd
import numpy as np

from transformers import AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns  # For more attractive plotting

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
 
import pandas as pd

import transformers
from transformers import logging

logging.set_verbosity_error()

import re
from IPython.display import display, Markdown

import markdown2
import pdfkit

 
import uuid
import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
import networkx as nx
import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader, PyPDFium2Loader
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import random
from pyvis.network import Network

from tqdm.notebook import tqdm

import seaborn as sns
palette = "hls"

import uuid
import pandas as pd
import numpy as np

from transformers import AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns  # For more attractive plotting

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Code based on: https://github.com/rahulnyk/knowledge_graph

def extract (string, start='[', end=']'):
    start_index = string.find(start)
    end_index = string.rfind(end)
     
    return string[start_index :end_index+1]
def documents2Dataframe(documents) -> pd.DataFrame:
    rows = []
    for chunk in documents:
        row = {
            "text": chunk,
           # **chunk.metadata,
            "chunk_id": uuid.uuid4().hex,
        }
        rows = rows + [row]

    df = pd.DataFrame(rows)
    return df

def concepts2Df(concepts_list) -> pd.DataFrame:
    ## Remove all NaN entities
    concepts_dataframe = pd.DataFrame(concepts_list).replace(" ", np.nan)
    concepts_dataframe = concepts_dataframe.dropna(subset=["entity"])
    concepts_dataframe["entity"] = concepts_dataframe["entity"].apply(
        lambda x: x.lower()
    )

    return concepts_dataframe


def df2Graph(dataframe: pd.DataFrame, generate, repeat_refine=0, verbatim=False,
          
            ) -> list:
  
    results = dataframe.apply(
        lambda row: graphPrompt(row.text, generate, {"chunk_id": row.chunk_id}, repeat_refine=repeat_refine,
                                verbatim=verbatim,#model
                               ), axis=1
    )
    # invalid json results in NaN
    results = results.dropna()
    results = results.reset_index(drop=True)

    ## Flatten the list of lists to one single list of entities.
    concept_list = np.concatenate(results).ravel().tolist()
    return concept_list


def graph2Df(nodes_list) -> pd.DataFrame:
    ## Remove all NaN entities
    graph_dataframe = pd.DataFrame(nodes_list).replace(" ", np.nan)
    graph_dataframe = graph_dataframe.dropna(subset=["node_1", "node_2"])
    graph_dataframe["node_1"] = graph_dataframe["node_1"].apply(lambda x: str(x).lower())
    graph_dataframe["node_2"] = graph_dataframe["node_2"].apply(lambda x: str(x).lower())

    return graph_dataframe

import sys
from yachalk import chalk
sys.path.append("..")

import json

# ---------------------------------------------------------------------------
# MARS MODIFICATION (see VENDORED.md)
# ---------------------------------------------------------------------------
# Upstream graphPrompt() asked the LLM for a bare JSON array of
# {node_1, node_2, edge} and treated generate()'s return value as a raw string
# (response.replace(...), extract(...), json.loads(...)). MARS's generate()
# returns a validated pydantic KnowledgeGraph object, so the very first
# .replace() raised AttributeError before any validation could run. Upstream
# also overrode the caller's system_prompt on every call, which made the prompt
# in data/KG_Generation/build_kg.py dead code.
#
# This version makes build_kg.py's schema authoritative: one call per chunk,
# asking for {"nodes": [{id, type}], "edges": [{source, target, relation}]},
# and it accepts a pydantic object, a dict, or a JSON string so either contract
# works. The RETURN shape is unchanged from upstream -- a list of
# {node_1, node_2, edge} dicts merged with `metadata` -- because graph2Df()
# drops any row missing the node_1/node_2 columns.

SYS_PROMPT_GRAPHMAKER = (
    "You are a scientific assistant extracting knowledge graphs from text.\n"
    "Return a JSON object with exactly two fields: \"nodes\" and \"edges\".\n"
    "Each node must have \"id\" (string) and \"type\" (string).\n"
    "Each edge must have \"source\" (string), \"target\" (string), and "
    "\"relation\" (string).\n"
    "Node ids must be concise, widely used terms from materials science.\n"
    "Every edge source and target must appear as a node id.\n"
    "Extract around 10 relationships.\n"
    "Example: {\"nodes\": [{\"id\": \"PEEK\", \"type\": \"material\"}], "
    "\"edges\": [{\"source\": \"PEEK\", \"target\": \"thermoplastic\", "
    "\"relation\": \"is_a\"}]}"
)


def _coerce_kg(response):
    """Accept a pydantic model, a dict, or a (possibly fenced) JSON string."""
    if response is None:
        return None
    if hasattr(response, "model_dump"):        # pydantic v2
        return response.model_dump()
    if hasattr(response, "dict") and not isinstance(response, dict):
        return response.dict()                 # pydantic v1
    if isinstance(response, dict):
        return response
    if isinstance(response, str):
        txt = response.strip()
        if txt.startswith("```"):              # strip ```json fences
            txt = re.sub(r"^```[a-zA-Z]*\n?", "", txt)
            txt = re.sub(r"```$", "", txt).strip()
        try:
            return json.loads(txt)
        except json.JSONDecodeError:
            lo, hi = txt.find("{"), txt.rfind("}")
            if lo != -1 and hi > lo:
                try:
                    return json.loads(txt[lo:hi + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _kg_to_triplets(kg, metadata):
    """{"nodes": [...], "edges": [...]} -> upstream's [{node_1, node_2, edge}]."""
    if not isinstance(kg, dict):
        return []
    triplets, seen = [], set()
    for e in kg.get("edges") or []:
        if not isinstance(e, dict):
            continue
        s = str(e.get("source", "")).strip()
        t = str(e.get("target", "")).strip()
        r = str(e.get("relation", "")).strip() or "related to"
        if not s or not t or s.lower() == t.lower():
            continue
        key = (s.lower(), t.lower(), r.lower())
        if key in seen:
            continue
        seen.add(key)
        triplets.append(dict({"node_1": s, "node_2": t, "edge": r}, **metadata))
    return triplets


def graphPrompt(input: str, generate, metadata={}, repeat_refine=0, verbatim=False):
    user_prompt = f"Context: ```{input}``` \n\nOutput: "
    print(".", end="")

    response = generate(system_prompt=SYS_PROMPT_GRAPHMAKER, prompt=user_prompt)
    if verbatim:
        print("---------------------\nFirst result: ", response)

    kg = _coerce_kg(response)
    if kg is None:
        print("\n\nERROR ### Could not parse response: ", response, "\n\n")
        return None

    triplets = _kg_to_triplets(kg, metadata)

    # Optional extra passes, mirroring upstream's repeat_refine contract.
    for rep in range(repeat_refine):
        refine_prompt = (
            f"Context: ```{input}```\n"
            f"Already extracted: ```{json.dumps(kg)[:4000]}```\n\n"
            "Add any relationships that are present in the context but missing "
            "from the extraction. Return the SAME JSON object format, containing "
            "both the original and the new nodes and edges."
        )
        response = generate(system_prompt=SYS_PROMPT_GRAPHMAKER, prompt=refine_prompt)
        more = _coerce_kg(response)
        if verbatim:
            print(f"---------------------\nAfter refine {rep}/{repeat_refine}: ", response)
        if more is None:
            break
        kg = more
        extra = _kg_to_triplets(more, metadata)
        known = {(t["node_1"].lower(), t["node_2"].lower(), t["edge"].lower())
                 for t in triplets}
        triplets += [t for t in extra
                     if (t["node_1"].lower(), t["node_2"].lower(), t["edge"].lower())
                     not in known]

    if not triplets:
        print("\n\nERROR ### No usable triplets in response: ", response, "\n\n")
        return None
    return triplets


def colors2Community(communities) -> pd.DataFrame:
    
    p = sns.color_palette(palette, len(communities)).as_hex()
    random.shuffle(p)
    rows = []
    group = 0
    for community in communities:
        color = p.pop()
        group += 1
        for node in community:
            rows += [{"node": node, "color": color, "group": group}]
    df_colors = pd.DataFrame(rows)
    return df_colors

def contextual_proximity(df: pd.DataFrame) -> pd.DataFrame:
    ## Melt the dataframe into a list of nodes
    df['node_1'] = df['node_1'].astype(str)
    df['node_2'] = df['node_2'].astype(str)
    df['edge'] = df['edge'].astype(str)
    dfg_long = pd.melt(
        df, id_vars=["chunk_id"], value_vars=["node_1", "node_2"], value_name="node"
    )
    dfg_long.drop(columns=["variable"], inplace=True)
    # Self join with chunk id as the key will create a link between terms occuring in the same text chunk.
    dfg_wide = pd.merge(dfg_long, dfg_long, on="chunk_id", suffixes=("_1", "_2"))
    # drop self loops
    self_loops_drop = dfg_wide[dfg_wide["node_1"] == dfg_wide["node_2"]].index
    dfg2 = dfg_wide.drop(index=self_loops_drop).reset_index(drop=True)
    ## Group and count edges.
    dfg2 = (
        dfg2.groupby(["node_1", "node_2"])
        .agg({"chunk_id": [",".join, "count"]})
        .reset_index()
    )
    dfg2.columns = ["node_1", "node_2", "chunk_id", "count"]
    dfg2.replace("", np.nan, inplace=True)
    dfg2.dropna(subset=["node_1", "node_2"], inplace=True)
    # Drop edges with 1 count
    dfg2 = dfg2[dfg2["count"] != 1]
    dfg2["edge"] = "contextual proximity"
    return dfg2
    
def make_graph_from_text (txt,generate,
                          generate_figure=None, image_list=None, do_distill=None,
                          include_contextual_proximity=False,
                          graph_root='graph_root',
                          chunk_size=2500,chunk_overlap=0,
                          repeat_refine=0,verbatim=False,
                          data_dir='./data_output_KG/',
                          save_PDF=False,#TO DO
                          save_HTML=True,
                         ):    
    
    # MARS MODIFICATION (see VENDORED.md): `generate_figure`, `image_list` and
    # `do_distill` existed in the GraphReasoning version MARS was written against
    # and were dropped upstream. They are accepted and ignored so that
    # data/KG_Generation/build_kg.py keeps working unchanged:
    #   - generate_figure: MARS passes a stub that returns an empty graph
    #   - image_list:      MARS passes '' (no figures)
    #   - do_distill:      MARS passes False (the pre-pass is off)
    # Note generate_figure is the 3rd POSITIONAL parameter because build_kg.py
    # passes it positionally.
    ## data directory
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)     
     
    outputdirectory = Path(f"./{data_dir}/") #where graphs are stored from graph2df function
    
 
    splitter = RecursiveCharacterTextSplitter(
        #chunk_size=5000, #1500,
        chunk_size=chunk_size, #1500,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    pages = splitter.split_text(txt)
    print("Number of chunks = ", len(pages))
    if verbatim:
        display(Markdown (pages[0]) )
    
    df = documents2Dataframe(pages)

    ## To regenerate the graph with LLM, set this to True
    regenerate = True
    
    if regenerate:
        concepts_list = df2Graph(df,generate,repeat_refine=repeat_refine,verbatim=verbatim) #model='zephyr:latest' )
        dfg1 = graph2Df(concepts_list)
        if not os.path.exists(outputdirectory):
            os.makedirs(outputdirectory)
        
        dfg1.to_csv(outputdirectory/f"{graph_root}_graph.csv", sep="|", index=False)
        df.to_csv(outputdirectory/f"{graph_root}_chunks.csv", sep="|", index=False)
        dfg1.to_csv(outputdirectory/f"{graph_root}_graph_clean.csv", #sep="|", index=False
                   )
        df.to_csv(outputdirectory/f"{graph_root}_chunks_clean.csv", #sep="|", index=False
                 )
    else:
        dfg1 = pd.read_csv(outputdirectory/f"{graph_root}_graph.csv", sep="|")
    
    dfg1.replace("", np.nan, inplace=True)
    dfg1.dropna(subset=["node_1", "node_2", 'edge'], inplace=True)
    dfg1['count'] = 4 
      
    if verbatim:
        print("Shape of graph DataFrame: ", dfg1.shape)
    dfg1.head()### 
    
    if include_contextual_proximity:
        dfg2 = contextual_proximity(dfg1)
        dfg = pd.concat([dfg1, dfg2], axis=0)
        #dfg2.tail()
    else:
        dfg=dfg1
        
    
    dfg = (
        dfg.groupby(["node_1", "node_2"])
        .agg({"chunk_id": ",".join, "edge": ','.join, 'count': 'sum'})
        .reset_index()
    )
    #dfg
        
    nodes = pd.concat([dfg['node_1'], dfg['node_2']], axis=0).unique()
    print ("Nodes shape: ", nodes.shape)
    
    G = nx.DiGraph()   # MARS MODIFICATION (see VENDORED.md): was nx.Graph()
    node_list=[]
    node_1_list=[]
    node_2_list=[]
    title_list=[]
    weight_list=[]
    chunk_id_list=[]
    
    ## Add nodes to the graph
    for node in nodes:
        G.add_node(
            str(node)
        )
        node_list.append (node)
    
    ## Add edges to the graph
    for _, row in dfg.iterrows():
        
        G.add_edge(
            str(row["node_1"]),
            str(row["node_2"]),
            title=row["edge"],
            weight=row['count']/4
        )
        
        node_1_list.append (row["node_1"])
        node_2_list.append (row["node_2"])
        title_list.append (row["edge"])
        weight_list.append (row['count']/4)
         
        chunk_id_list.append (row['chunk_id'] )

    try:
            
        df_nodes = pd.DataFrame({"nodes": node_list} )    
        df_nodes.to_csv(f'{data_dir}/{graph_root}_nodes.csv')
        df_nodes.to_json(f'{data_dir}/{graph_root}_nodes.json')
        
        df_edges = pd.DataFrame({"node_1": node_1_list, "node_2": node_2_list,"edge_list": title_list, "weight_list": weight_list } )    
        df_edges.to_csv(f'{data_dir}/{graph_root}_edges.csv')
        df_edges.to_json(f'{data_dir}/{graph_root}_edges.json')
        
    except:
        
        print ("Error saving CSV/JSON files.")
    
    communities_generator = nx.community.girvan_newman(G)
    #top_level_communities = next(communities_generator)
    next_level_communities = next(communities_generator)
    communities = sorted(map(sorted, next_level_communities))
    
    if verbatim:
        print("Number of Communities = ", len(communities))
        
    if verbatim:
        print("Communities: ", communities)
    
    colors = colors2Community(communities)
    if verbatim:
        print ("Colors: ", colors)
    
    for index, row in colors.iterrows():
        G.nodes[row['node']]['group'] = row['group']
        G.nodes[row['node']]['color'] = row['color']
        G.nodes[row['node']]['size'] = G.degree[row['node']]
            
    net = Network(
             
            notebook=True,
         
            cdn_resources="remote",
            height="900px",
            width="100%",
            select_menu=True,
            
            filter_menu=False,
        )
        
    net.from_nx(G)
    net.force_atlas_2based(central_gravity=0.015, gravity=-31)
   
    net.show_buttons()
    
    graph_HTML= f'{data_dir}/{graph_root}_grapHTML.html'
    graph_GraphML=  f'{data_dir}/{graph_root}_graphML.graphml'  #  f'{data_dir}/resulting_graph.graphml',
    nx.write_graphml(G, graph_GraphML)
    
    if save_HTML:
        net.show(graph_HTML,
            )

    if save_PDF:
        output_pdf=f'{data_dir}/{graph_root}_PDF.pdf'
        pdfkit.from_file(graph_HTML,  output_pdf)
    else:
        output_pdf=None
    res_stat=graph_statistics_and_plots_for_large_graphs(G, data_dir=data_dir,include_centrality=False,
                                                       make_graph_plot=False,)
        
    print ("Graph statistics: ", res_stat)
    return graph_HTML, graph_GraphML, G, net, output_pdf

import time
from copy import deepcopy

def add_new_subgraph_from_text(txt,generate=None,node_embeddings=None,tokenizer=None, model=None,
                               original_graph_path_and_fname=None,
                               original_graph=None, graph_root='graph_root',
                               do_update_node_embeddings=None,
                               data_dir_output='./data_temp/', verbatim=True,
                               size_threshold=10,chunk_size=10000,
                               do_Louvain_on_new_graph=True,include_contextual_proximity=False,repeat_refine=0,similarity_threshold=0.95, do_simplify_graph=True,#whether or not to simplify, uses similiraty_threshold defined above
                               return_only_giant_component=False,
                               save_common_graph=False,G_to_add=None,graph_GraphML_to_add=None,
                              ):

    display (Markdown(txt[:256]+"...."))
    graph_GraphML=None
     
    G_new=None
    res=None
    assert not (G_to_add is not None and graph_GraphML_to_add is not None), "G_to_add and graph_GraphML_to_add cannot be used together. Pick one or the other to provide a graph to be added."
 
    try:
        start_time = time.time() 
        idx=0
        
        if verbatim:
            print ("Now create or load new graph...")

        # MARS MODIFICATION (see VENDORED.md): upstream referenced `G_newlymade`,
        # a name defined nowhere in the package, so this line always raised
        # NameError. The bare `except` below swallowed it and printed
        # "ALERT: Graph generation failed", making a working merge look like a
        # failure. The two graph inputs are `graph_GraphML_to_add` and
        # `G_to_add`, so the latter is what the comment's condition means.
        if graph_GraphML_to_add is None and G_to_add is None: #make new if no existing one provided
            print ("Make new graph from text...")
            _, graph_GraphML_to_add, G_to_add, _, _ =make_graph_from_text (txt,generate,
                                      include_contextual_proximity=include_contextual_proximity,
                                      
                                     data_dir=data_dir_output,
                                     graph_root=f'graph_new_{idx}',
                                    
                                        chunk_size=chunk_size,   repeat_refine=repeat_refine, 
                                      verbatim=verbatim,
                                       
                                  )
            if verbatim:
                print ("Generated new graph from text provided: ", graph_GraphML_to_add)

        else:
            if verbatim:
                print ("Instead of generating graph, loading it or using provided graph...(any txt data provided will be ignored...)")

            if graph_GraphML_to_add!=None:
                print ("Loading graph: ", graph_GraphML_to_add)
        
        print("--- %s seconds ---" % (time.time() - start_time))
    except:
        print ("ALERT: Graph generation failed...for idx=",idx)
    
    print ("Now add node to existing graph...")
    
    try:
        # MARS MODIFICATION (see VENDORED.md): accept a live networkx graph via
        # `original_graph` in addition to upstream's path argument. build_kg.py
        # keeps the merged graph in memory across documents and would otherwise
        # have to round-trip it through disk on every iteration.
        if original_graph is not None:
            G = original_graph
        else:
            G = nx.read_graphml(original_graph_path_and_fname)
        
        if G_to_add!=None:
            G_loaded=H = deepcopy(G_to_add)
            if verbatim:
                print ("Using provided graph to add (any txt data provided will be ignored...)")
        else:
            if verbatim:
                print ("Loading graph to be added either newly generated or provided.")
            G_loaded = nx.read_graphml(graph_GraphML_to_add)
        
        res_newgraph=graph_statistics_and_plots_for_large_graphs(G_loaded, data_dir=data_dir_output,include_centrality=False,
                                                       make_graph_plot=False,root='new_graph')
        print (res_newgraph)
        
        G_new = nx.compose(G,G_loaded)

        if save_common_graph:
            print ("Identify common nodes and save...")
            try:
                
                common_nodes = set(G.nodes()).intersection(set(G_loaded.nodes()))
    
                subgraph = G_new.subgraph(common_nodes)
                graph_GraphML=  f'{data_dir_output}/{graph_root}_common_nodes_before_simple.graphml' 
                nx.write_graphml(subgraph, graph_GraphML)
            except: 
                print ("Common nodes identification failed.")
            print ("Done!")
        
        if verbatim:
            print ("Now update node embeddings")
        node_embeddings=update_node_embeddings(node_embeddings, G_new, tokenizer, model)
        print ("Done update node embeddings.")
        if do_simplify_graph:
            if verbatim:
                print ("Now simplify graph.")
            G_new, node_embeddings =simplify_graph (G_new, node_embeddings, tokenizer, model , 
                                                    similarity_threshold=similarity_threshold, use_llm=False, data_dir_output=data_dir_output,
                                    verbatim=verbatim,)
            if verbatim:
                print ("Done simplify graph.")
            
        if verbatim:
            print ("Done update graph")
        
        if size_threshold >0:
            if verbatim:
                print ("Remove small fragments")            
            G_new=remove_small_fragents (G_new, size_threshold=size_threshold)
            node_embeddings=update_node_embeddings(node_embeddings, G_new, tokenizer, model, verbatim=verbatim)
        
        if return_only_giant_component:
            if verbatim:
                print ("Select only giant component...")   
            connected_components = sorted(nx.connected_components(G_new.to_undirected() if G_new.is_directed() else G_new), key=len, reverse=True)
            G_new = G_new.subgraph(connected_components[0]).copy()
            node_embeddings=update_node_embeddings(node_embeddings, G_new, tokenizer, model, verbatim=verbatim)
            
        print (".")
        if do_Louvain_on_new_graph:
            G_new=graph_Louvain (G_new, 
                      graph_GraphML=None)
            if verbatim:
                print ("Don Louvain...")

        print (".")
         
        graph_root=f'graph'
        graph_GraphML=  f'{data_dir_output}/{graph_root}_augmented_graphML_integrated.graphml'  #  f'{data_dir}/resulting_graph.graphml',
        print (".")
        nx.write_graphml(G_new, graph_GraphML)
        print ("Done...written: ", graph_GraphML)
        res=graph_statistics_and_plots_for_large_graphs(G_new, data_dir=data_dir_output,include_centrality=False,
                                                       make_graph_plot=False,root='assembled')
        
        print ("Graph statistics: ", res)

    except Exception as _e:
        # MARS MODIFICATION (see VENDORED.md): upstream swallowed this with a bare
        # `except:` and a one-line message, which hid real failures (wrong argument
        # names, arity changes) behind "Error adding new graph." Print the traceback.
        import traceback as _tb
        print ("Error adding new graph:", type(_e).__name__, _e)
        _tb.print_exc()
        print (end="")

    return graph_GraphML, G_new, G_loaded, G, node_embeddings, res
