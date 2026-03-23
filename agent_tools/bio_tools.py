import pandas as pd
from .env_utils import get_medeadb_path as _get_medeadb_path
import unicodedata
import requests
import difflib
import torch 
import json 
import os
import copy
import numpy as np
import os
import h5py
from typing import Dict, Tuple, Optional, Union, List, Any

import urllib.parse
import networkx as nx
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import Counter
import xml.etree.ElementTree as ET
import logging
import glob



from .env_utils import get_medeadb_path as _get_medeadb_path

base_url = "https://api.platform.opentargets.org/api/v4/graphql"

#### Action Functions ####
open_target_query_string = """
    query target($diseaseName: String!){
        search(queryString: $diseaseName){
        hits{
            id
            entity
            category
            name
            description
        }
    }
}
"""

QUERY_DISEASEID = """Please provide the disease ID of {disease_name} from OpenTargets using the format: 'DiseaseID of <disease>: <diseaseID>'.
---
Here are related disease entites from OpenTargets:
{disease_entities}
"""


def build_pinnacle_ppi(ppi_embed_path: str, labels_path: str):
    """Load the PINNACLE celltype-specific PPI net with embeddings for each gene on different cell types.

    Args:
        ppi_embed_path (str): path to the PINNACLE PPI embeddings
        labels_path (str): path to the labels file containing the cell type and activated genes for each cell type
    
    Return:
        ppi_embed_dict (dict): a dictionary containing the cell type as key, and the value is a celltype-specific PPI dict with gene embeddings. For celltype-specific PPI dict, the key is the gene name and the value is the gene embedding.
        
    """
    embed = torch.load(ppi_embed_path)
    with open(labels_path, "r") as f:
        labels_dict = f.read()
    labels_dict = labels_dict.replace("\'", "\"")
    labels_dict = json.loads(labels_dict)
    celltypes = [c for c in labels_dict["Cell Type"] if c.startswith("CCI")]
    celltype_dict = {ct.split("CCI_")[1]: i for i, ct in enumerate(celltypes)}
    assert len(celltype_dict) > 0
    
    protein_names = []
    protein_celltypes = []
    for c, p in zip(labels_dict["Cell Type"], labels_dict["Name"]):
        if c.startswith("BTO") or c.startswith("CCI") or c.startswith("Sanity"): continue
        protein_names.append(p) 
        protein_celltypes.append(c) 

    proteins = pd.DataFrame.from_dict({"target": protein_names, "cell type": protein_celltypes})
    celltype_protein_dict = proteins.pivot_table(values="target", index="cell type", aggfunc={"target": list}).to_dict()["target"]
    assert len(celltype_protein_dict) > 0
    
    ppi_embed_dict = {}
    for celltype, index in celltype_dict.items():
        cell_embed_dict = {}
        cell_embed = embed[index]
        for i, gene in enumerate(celltype_protein_dict[celltype]):
            gene_embed = cell_embed[i, :]
            cell_embed_dict[gene] = gene_embed
            # print(f"[pinnacle]: {celltype} - {gene} - {gene_embed.shape}")
        celltype = celltype.replace(" ", "_")
        ppi_embed_dict[celltype] = cell_embed_dict
    return ppi_embed_dict


def load_pinnacle_ppi(
    cell_type: str,
    embed_path: str = None, 
    weights_only=False
):
    """
    Load the PINNACLE PPI embeddings for a specific cell type from a specified path.
    
    Uses the same normalization and matching logic as celltype_avaliability_checker() to ensure consistency.
    Accepts multiple cell type formats and normalizes them automatically.
    
    Args:
        cell_type (str): The specific cell type to load embeddings for. Accepts multiple formats:
                        - Standardized: 'cd4_positive_alpha_beta_memory_t_cell'
                        - Raw: 'cd4-positive,_alpha-beta_memory_t_cell'
                        - User-friendly: 'CD4+ memory T cell'
                        All formats are normalized to the same internal representation.
        embed_path (str): Path to the PPI embeddings file. If None, uses MEDEADB_PATH environment variable.
        weights_only (bool): Whether to load weights only or the full state.
    
    Returns:
        dict: A dictionary of PPI embeddings for the specified cell type with gene name as key 
            and cell type-specific gene embedding as value (torch.Tensor). Returns empty dict if cell type not found.
    """
    # Set default embed_path if not provided
    if embed_path is None:
        embed_path = os.path.join(_get_medeadb_path(), 'pinnacle_embeds/ppi_embed_dict.pth')
    
    # Load the full PPI dictionary from the specified path
    ppi_dict = torch.load(embed_path, weights_only=weights_only)
    
    # Normalize cell type using same function as checker
    def _normalize(s):
        return s.replace(",", "").replace("-", "_").replace(" ", "_").replace("+", "_positive").replace("α", "alpha").replace("β", "beta").lower()
    
    def _format_display(s):
        """Format for clean, consistent display."""
        formatted = s.replace(",", "").replace("-", "_").replace(" ", "_")
        while "__" in formatted:
            formatted = formatted.replace("__", "_")
        return formatted.lower()
    
    formalized_cell_type = _normalize(cell_type)
    
    # First pass: exact match
    for cell_key in ppi_dict.keys():
        formalized_key = _normalize(cell_key)
        if formalized_key == formalized_cell_type:
            formatted_name = _format_display(cell_key)
            print(f"[load_pinnacle_ppi] ✓ Loaded {len(ppi_dict[cell_key])} genes for cell type: '{formatted_name}'", flush=True)
            return ppi_dict[cell_key]
        
    # Second pass: fuzzy matching with same logic as checker
    from thefuzz import fuzz
    best_match = None
    best_score = 0
    
    for cell_key in ppi_dict.keys():
        candidate_norm = _normalize(cell_key)
        
        # Compute similarity score using multiple strategies
        token_sort = fuzz.token_sort_ratio(formalized_cell_type, candidate_norm)
        token_set = fuzz.token_set_ratio(formalized_cell_type, candidate_norm)
        
        # Combined score
        score = token_sort * 0.7 + token_set * 0.3
        
        # Token overlap boost
        query_tokens = set(formalized_cell_type.split('_'))
        candidate_tokens = set(candidate_norm.split('_'))
        if query_tokens and candidate_tokens:
            intersection = len(query_tokens.intersection(candidate_tokens))
            union = len(query_tokens.union(candidate_tokens))
            jaccard = intersection / union if union > 0 else 0
            score += jaccard * 15
        
        if score > best_score:
            best_score = score
            best_match = cell_key
    
    # Return best match if score is reasonable
    if best_match and best_score >= 60:
        formatted_name = _format_display(best_match)
        print(f"[load_pinnacle_ppi] ⚠ Cell type '{cell_type}' matched to '{formatted_name}' (score: {best_score:.0f})", flush=True)
        print(f"[load_pinnacle_ppi] ✓ Loaded {len(ppi_dict[best_match])} genes for matched cell type", flush=True)
        return ppi_dict[best_match]
    
    # If no good match found, return empty dict
    print(f"[load_pinnacle_ppi] ✗ ERROR: Cell type '{cell_type}' not found in PINNACLE embeddings (best match score: {best_score:.0f}).", flush=True)
    print(f"[load_pinnacle_ppi] → Please use the celltype_avaliability_checker to find valid cell types.", flush=True)
    return {}


def read_labels_from_evidence(
        positive_protein_prefix, 
        negative_protein_prefix, 
        raw_data_prefix, 
        positive_proteins={}, 
        negative_proteins={}, 
        all_relevant_proteins={}
    ):
    try:
        with open(positive_protein_prefix + '.json', 'r') as f:
            temp = json.load(f)
            positive_proteins = temp
        with open(negative_protein_prefix + '.json', 'r') as f:
            temp = json.load(f)
            negative_proteins = temp
        
        if raw_data_prefix != None:
            with open(raw_data_prefix + '.json', 'r') as f:
                temp = json.load(f)
                all_relevant_proteins = temp
        else: all_relevant_proteins = {}

        return positive_proteins, negative_proteins, all_relevant_proteins
    except:
        print("Files not found")
        return {}, {}, {}


def search_disease_open_target(disease_name):
    # Set variables object of arguments to be passed to endpoint
    relavent_entities = None
    variables = {"diseaseName": disease_name}
    for i in range(5):
        try:
            # Perform POST request and check status code of response
            r = requests.post(base_url, json={"query": open_target_query_string, "variables": variables})
            # Transform API response from JSON into Python dictionary and print in console
            api_response = json.loads(r.text)
            relavent_entities = api_response['data']['search']['hits']
        except Exception as e:
            print(f"[OpenTarget] Bad OpenTarget API response, retrying...")
            continue
        if relavent_entities is not None: break

    if relavent_entities is None:
        raise ValueError(f"[OpenTarget] No relevant entities found for {disease_name}")
    # Check the top 5 entities
    filtered_entities = [ e for e in relavent_entities if e['entity'] == 'disease']
    return filtered_entities


import requests
import re

def normalize_string(s):
    """
    Normalize a string by removing punctuation and converting to lowercase.
    """
    return re.sub(r'[^\w\s]', '', s).lower().strip()



def standardize_disease_name(disease_name):
    # Normalize unicode characters to remove accents
    disease_name = unicodedata.normalize('NFKD', disease_name).encode('ASCII', 'ignore').decode('utf-8')
    
    # Remove possessive apostrophes (e.g., "'s")
    disease_name = re.sub(r"'s\b", "", disease_name)
    
    # Remove any remaining punctuation
    disease_name = re.sub(r"[^\w\s]", "", disease_name)
    
    # Normalize whitespace
    disease_name = re.sub(r"\s+", " ", disease_name).strip()
    
    # Capitalize each word (optional, depending on your target style)
    disease_name = disease_name.title()
    
    return disease_name


def get_efo_id(disease_name):
    """
    Retrieve the EFO identifier for a given disease name using the EMBL-EBI OLS API.

    Args:
        disease_name (str): The name of the disease.

    Returns:
        str: The EFO identifier if found, else None.
    """
    # Define the API endpoint
    api_url = "https://www.ebi.ac.uk/ols/api/search"
    disease_name = standardize_disease_name(disease_name)

    # First attempt: exact match search
    params = {
        'q': disease_name,
        'ontology': 'efo',
        'exact': 'true'
    }
    response = requests.get(api_url, params=params)
    
    if response.status_code == 200:
        results = response.json()
        if results['response']['numFound'] > 0:
            return results['response']['docs'][0]['obo_id']
    else:
        print(f"Failed to connect to OLS API. Status code: {response.status_code}")
        return None

    # If exact match not found, try fuzzy search with normalized labels.
    print(f"Exact match not found for '{disease_name}'. Trying fuzzy search.")
    params['exact'] = 'false'
    response = requests.get(api_url, params=params)
    
    if response.status_code == 200:
        results = response.json()
        normalized_target = normalize_string(disease_name)
        for doc in results['response']['docs']:
            label = doc.get('label', '')
            if normalize_string(label) == normalized_target:
                return doc.get('obo_id')
        print(f"No matching EFO ID found for '{disease_name}' in fuzzy search.")
        return None
    else:
        print(f"Failed to connect to OLS API on fuzzy search. Status code: {response.status_code}")
        return None

def search_disease_efo(disease_name, attempts=5):
    """
    Searches for the EFO ID of a given disease in the EFO ontology.
    
    Args:
        disease_name (str): The name of the disease to search for.
        
    Returns:
        str: The EFO or MONDO ID of the disease if found, otherwise None.
    """
    
    # Attempt to query up to 5 times to handle potential issues
    for attempt in range(attempts):
        try:
            # Search for disease entities using the Open Targets platform
            disease_name = disease_name.replace("_", " ")
            disease_efo = get_efo_id(disease_name)
            disease_id = disease_efo.replace(":", "_")
            
            # Check if the response contains a valid EFO or MONDO ID
            if 'EFO' in disease_id or 'MONDO' in disease_id:
                return disease_id
        except Exception as e:
            print(f"[Attempt {attempt + 1}/5] Error: {e}")
            print("[search_disease_efo]: Bad request format or other issue encountered. Retrying...")
    
    # Return None if a valid ID was not found after 5 attempts
    print(f"[search_disease_efo]: Failed to find EFO or MONDO ID for '{disease_name}' after {attempts} attempts.")
    return None


def compare_strings(str1, str2):
    matcher = difflib.SequenceMatcher(None, str1, str2)
    return matcher.ratio()


def load_disease_targets(disease_name, data_dir=None, attributes=["otGeneticsPortal", "chembl"], use_api=True, max_retries=5):
    """
    Load the disease-associated targets from OpenTargets API or local JSON file.

    Parameters:
        disease_name (str): The name of the disease.
        data_dir (str): The directory containing disease target data. Only used if use_api=False.
        attributes (list): List of attributes to filter targets. 
                          For API: filters by datasource types (e.g., "ot_genetics_portal", "chembl")
                          For local: filters by JSON field names (e.g., "otGeneticsPortal", "chembl")
        use_api (bool): If True, retrieve data from OpenTargets API. If False, use local JSON files.
        max_retries (int): Maximum number of retry attempts for API calls.

    Returns:
        set: A set of gene symbols associated with the disease.
    """
    if use_api:
        return _load_disease_targets_from_api(disease_name, attributes, max_retries)
    else:
        return _load_disease_targets_from_local(disease_name, data_dir, attributes)


def _load_disease_targets_from_api(disease_name, attributes=["otGeneticsPortal", "chembl"], max_retries=5):
    """
    Load disease-associated targets from OpenTargets GraphQL API.
    
    Parameters:
        disease_name (str): The name of the disease.
        attributes (list): List of datasource types to filter targets.
        max_retries (int): Maximum number of retry attempts.
    
    Returns:
        set: A set of gene symbols associated with the disease.
    """
    # Get the EFO/MONDO ID for the disease
    disease_efo = search_disease_efo(disease_name)
    if disease_efo is None:
        raise ValueError(f"[load_disease_targets] Could not find EFO/MONDO ID for disease: {disease_name}")
    
    print(f"[load_disease_targets] Querying OpenTargets API for disease: {disease_name} ({disease_efo})")
    
    # Convert attribute names to match OpenTargets datasource format
    # Map common attribute names to OpenTargets datasource IDs
    # Actual API datasource IDs: genetic_association, genetic_literature, known_drug, 
    # literature, rna_expression, animal_model, somatic_mutation, etc.
    datasource_mapping = {
        "otGeneticsPortal": "genetic_association",
        "chembl": "known_drug",
        "europepmc": "literature",
        "expression_atlas": "rna_expression",
        "intogen": "somatic_mutation",
        "literature": "literature",
        "genetic_association": "genetic_association",
        "genetic_literature": "genetic_literature",
        "known_drug": "known_drug",
        "animal_model": "animal_model",
        "rna_expression": "rna_expression",
        "somatic_mutation": "somatic_mutation"
    }
    
    # Normalize attributes
    normalized_attributes = []
    for attr in attributes:
        if attr in datasource_mapping:
            normalized_attributes.append(datasource_mapping[attr])
        else:
            normalized_attributes.append(attr.lower())
    
    # GraphQL query to get disease-target associations
    query = """
    query diseaseTargets($efoId: String!, $size: Int!, $index: Int!) {
        disease(efoId: $efoId) {
            id
            name
            associatedTargets(page: {size: $size, index: $index}) {
                count
                rows {
                    target {
                        id
                        approvedSymbol
                    }
                    score
                    datatypeScores {
                        id
                        score
                    }
                }
            }
        }
    }
    """
    
    # Implement pagination (API max page size is 3000)
    page_size = 3000
    page_index = 0
    target_set = set()
    total_count = None
    
    while True:
        # Use the EFO ID as-is (with underscores, not colons)
        variables = {
            "efoId": disease_efo,
            "size": page_size,
            "index": page_index
        }
        
        # Retry logic for API calls
        success = False
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    base_url,
                    json={"query": query, "variables": variables},
                    timeout=30
                )
                
                # Check response before raising for status to see error details
                if response.status_code != 200:
                    try:
                        error_detail = response.json()
                        print(f"[load_disease_targets] API Error Detail: {json.dumps(error_detail, indent=2)}")
                    except:
                        print(f"[load_disease_targets] API Error: {response.text}")
                
                response.raise_for_status()
                
                api_response = response.json()
                
                # Check for errors in the GraphQL response
                if "errors" in api_response:
                    print(f"[load_disease_targets] GraphQL errors: {api_response['errors']}")
                    raise ValueError(f"GraphQL query returned errors: {api_response['errors']}")
                
                # Extract disease data
                disease_data = api_response.get("data", {}).get("disease")
                if not disease_data:
                    raise ValueError(f"[load_disease_targets] No disease data found for {disease_name} ({disease_efo})")
                
                associated_targets = disease_data.get("associatedTargets", {})
                rows = associated_targets.get("rows", [])
                total_count = associated_targets.get("count", 0)
                
                if page_index == 0:
                    print(f"[load_disease_targets] Found {total_count} total target associations from API")
                
                # Filter targets based on attributes if specified
                if attributes and len(attributes) > 0:
                    for row in rows:
                        datatype_scores = row.get("datatypeScores", [])
                        # Check if any of the specified datasources have a score
                        has_relevant_datasource = False
                        for ds in datatype_scores:
                            datasource_id = ds.get("id", "").lower()
                            score = ds.get("score", 0)
                            if any(attr in datasource_id for attr in normalized_attributes) and score > 0:
                                has_relevant_datasource = True
                                break
                        
                        if has_relevant_datasource:
                            symbol = row.get("target", {}).get("approvedSymbol")
                            if symbol:
                                target_set.add(symbol)
                else:
                    # If no attributes specified, return all targets with any association
                    for row in rows:
                        symbol = row.get("target", {}).get("approvedSymbol")
                        if symbol and row.get("score", 0) > 0:
                            target_set.add(symbol)
                
                success = True
                break
                
            except requests.exceptions.RequestException as e:
                print(f"[load_disease_targets] Attempt {attempt + 1}/{max_retries} - Network error: {e}")
                if attempt == max_retries - 1:
                    raise ValueError(f"[load_disease_targets] Failed to retrieve data from OpenTargets API after {max_retries} attempts: {e}")
            except Exception as e:
                print(f"[load_disease_targets] Attempt {attempt + 1}/{max_retries} - Error: {e}")
                if attempt == max_retries - 1:
                    raise ValueError(f"[load_disease_targets] Failed to process API response after {max_retries} attempts: {e}")
        
        if not success:
            break
        
        # Check if we've fetched all pages
        if len(rows) < page_size or (page_index + 1) * page_size >= total_count:
            break
        
        page_index += 1
        print(f"[load_disease_targets] Fetching page {page_index + 1} (retrieved {len(target_set)} targets so far)...")
    
    print(f"[load_disease_targets] Retrieved {len(target_set)} targets from OpenTargets API after filtering")
    return target_set


def _load_disease_targets_from_local(disease_name, data_dir=None, attributes=["otGeneticsPortal", "chembl"]):
    """
    Load disease-associated targets from local JSON file (legacy method).
    
    Parameters:
        disease_name (str): The name of the disease.
        data_dir (str): The directory containing disease target data.
        attributes (list): List of attributes to filter targets.
    
    Returns:
        set: A set of gene symbols associated with the disease.
    """
    # Set default data_dir if not provided
    if data_dir is None:
        data_dir = os.path.join(_get_medeadb_path(), "targetID/disease_target")
    
    critera_flag, target_set = False, set()
    disease_efo = search_disease_efo(disease_name)
    file_path = os.path.join(data_dir, disease_efo + '.json')
    if not os.path.exists(file_path):
        avaliable_diseases = []
        for _, dirnames, _ in os.walk(data_dir):
            for dir_name in dirnames:
                avaliable_diseases.append(dir_name)
        raise ValueError(f"[load_disease_targets] Disease: {disease_name} ({disease_efo}) is not avaliable locally, the avaliable disease options are: {avaliable_diseases}.")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
        for entry in data:
            for a in attributes:
                if entry.get(a, "No data") != "No data":
                    critera_flag = True
            if critera_flag:
                target_set.add(entry['symbol'])
    return target_set



def get_gene_synonyms(gene_name, species="Homo sapiens"):
    """
    Fetch synonyms for a given gene using the NCBI Entrez API.

    Args:
        gene_name (str): The name of the gene to search for.
        species (str): The species to filter results. Default is 'Homo sapiens'.

    Returns:
        list: A list of synonyms for the gene, or an error message if not found.
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

    # Step 1: Search for the gene in the NCBI Gene database
    search_params = {
        "db": "gene",
        "term": f"{gene_name}[Gene] AND {species}[Organism]",
        "retmode": "json"
    }
    while True:
        response = requests.get(base_url, params=search_params)
        if response.status_code == 200:
            break
        # print(f"Error: Unable to fetch data from NCBI. Status code: {response.status_code}")
        
    search_data = response.json()
    if "esearchresult" not in search_data or not search_data["esearchresult"].get("idlist"):
        return None

    gene_id = search_data["esearchresult"]["idlist"][0]  # Get the first gene ID

    # Step 2: Fetch gene details using the gene ID
    summary_params = {
        "db": "gene",
        "id": gene_id,
        "retmode": "json"
    }
    while True:
        summary_response = requests.get(summary_url, params=summary_params)
        if summary_response.status_code == 200:
            break
        # print( f"Error: Unable to fetch gene details. Status code: {summary_response.status_code}")

    summary_data = summary_response.json()
    gene_summary = summary_data.get("result", {}).get(gene_id, {})
    synonyms = gene_summary.get("otheraliases", "")

    # Return the synonyms as a list
    if synonyms:
        return synonyms.strip("'").split(", ")
    else:
        return None


#### Depmap Tools ####
class GeneCorrelationLookup:
    """
    Efficient lookup tool for gene-gene correlations and p-values from preprocessed data.
    
    This class provides fast access to Pearson correlation coefficients between 
    gene effect signatures derived from CERES scores across pan-cancer cell lines.
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize the lookup tool by loading the preprocessed gene correlation data.
        
        Parameters:
        -----------
        data_dir : str
            Path to the directory containing the preprocessed gene correlation data
            (should contain gene_names.txt and correlation matrix files)
        """
        self.data_dir = data_dir
        
        # Load data based on available file format (dense npy or sparse h5)
        self._load_gene_index()
        self._load_correlation_data()
        
    def _load_gene_index(self):
        """Load gene names and create mapping for fast lookups"""
        try:
            # Try loading from numpy array first (faster)
            gene_idx_path = os.path.join(self.data_dir, "gene_idx_array.npy")
            if os.path.exists(gene_idx_path):
                self.gene_names = np.load(gene_idx_path, allow_pickle=True, mmap_mode='r')
            else:
                # Fall back to text file
                gene_names_path = os.path.join(self.data_dir, "gene_names.txt")
                with open(gene_names_path, 'r') as f:
                    self.gene_names = np.array([line.strip() for line in f])
            
            # Create fast lookup dictionary
            self.gene_to_idx = {gene: idx for idx, gene in enumerate(self.gene_names)}
            self.num_genes = len(self.gene_names)
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Gene names file not found in {self.data_dir}")
    
    def _load_correlation_data(self):
        """Load correlation and p-value matrices based on available format"""
        # Check for dense matrix format first
        corr_matrix_path = os.path.join(self.data_dir, "corr_matrix.npy")
        p_val_matrix_path = os.path.join(self.data_dir, "p_val_matrix.npy")
        
        if os.path.exists(corr_matrix_path) and os.path.exists(p_val_matrix_path):
            # Dense matrix format
            self.corr_matrix = np.load(corr_matrix_path, mmap_mode='r')
            self.p_val_matrix = np.load(p_val_matrix_path, mmap_mode='r')
            self.format = "dense"
            
            # Check for adjusted p-values (optional)
            p_adj_matrix_path = os.path.join(self.data_dir, "p_adj_matrix.npy")
            if os.path.exists(p_adj_matrix_path):
                self.p_adj_matrix = np.load(p_adj_matrix_path, mmap_mode='r')
                self.has_adj_p = True
            else:
                self.has_adj_p = False
                
        else:
            # Try sparse HDF5 format
            h5_path = os.path.join(self.data_dir, "gene_correlations.h5")
            if not os.path.exists(h5_path):
                raise FileNotFoundError(f"No correlation data found in {self.data_dir}")
            
            # Load sparse matrices
            self.h5_file = h5py.File(h5_path, 'r')
            self.format = "sparse"
            
            # Check for adjusted p-values
            self.has_adj_p = 'p_adj' in self.h5_file
    
    def get_correlation(self, gene_a: str, gene_b: str) -> Dict[str, float]:
        """
        Get correlation coefficient and p-value between two genes.
        
        Parameters:
        -----------
        gene_a : str
            First gene symbol
        gene_b : str
            Second gene symbol
            
        Returns:
        --------
        dict
            Dictionary with 'correlation' and 'p_value' keys. If available,
            also includes 'adjusted_p_value'.
        
        Raises:
        -------
        KeyError
            If either gene is not found in the dataset
        """
        # Check if genes exist in our dataset
        if gene_a not in self.gene_to_idx:
            raise KeyError(f"Gene '{gene_a}' not avaliable in the gene correlation matrix")
        if gene_b not in self.gene_to_idx:
            raise KeyError(f"Gene '{gene_b}' not avaliable in the gene correlation matrix")
        
        # Get indices for the genes
        idx_a = self.gene_to_idx[gene_a]
        idx_b = self.gene_to_idx[gene_b]
        
        # Get values based on storage format
        if self.format == "dense":
            correlation = float(self.corr_matrix[idx_a, idx_b])
            p_value = float(self.p_val_matrix[idx_a, idx_b])
            
            result = {
                "correlation": correlation,
                "p_value": p_value
            }
            
            if self.has_adj_p:
                result["adjusted_p_value"] = float(self.p_adj_matrix[idx_a, idx_b])
                
        else:  # sparse format
            # Access sparse data from HDF5 file
            corr_data = self.h5_file['corr']
            p_val_data = self.h5_file['p_val']
            
            # We need to reconstruct the CSR matrix access pattern
            def get_csr_value(group, row, col):
                indptr = group['indptr'][:]
                indices = group['indices'][:]
                data = group['data'][:]
                
                # CSR lookup: check elements between indptr[row] and indptr[row+1]
                for i in range(indptr[row], indptr[row+1]):
                    if indices[i] == col:
                        return float(data[i])
                return 0.0  # If not found, assume zero (sparse matrix default)
            
            correlation = get_csr_value(corr_data, idx_a, idx_b)
            p_value = get_csr_value(p_val_data, idx_a, idx_b)
            
            result = {
                "correlation": correlation,
                "p_value": p_value
            }
            
            if self.has_adj_p:
                p_adj_data = self.h5_file['p_adj']
                result["adjusted_p_value"] = get_csr_value(p_adj_data, idx_a, idx_b)
        
        return result
    
    def get_cell_viability_effect(self, gene_a: str, gene_b: str) -> Dict[str, Union[float, str]]:
        """
        Get the interpreted cell viability effect based on correlation between two genes.
        
        Parameters:
        -----------
        gene_a : str
            First gene symbol
        gene_b : str
            Second gene symbol
            
        Returns:
        --------
        dict
            Dictionary with correlation statistics and interpretation of the relationship
            between the two genes in terms of cell viability effects.
        """
        corr_data = self.get_correlation(gene_a, gene_b)
        correlation = corr_data["correlation"]
        p_value = corr_data["p_value"]
        
        # Determine interaction interpretation
        if p_value > 0.05:
            interaction = "No statistically significant relationship"
        else:
            if correlation > 0.7:
                interaction = "Strong similar effect on cell viability"
            elif correlation > 0.5:
                interaction = "Moderate similar effect on cell viability"
            elif correlation > 0.3:
                interaction = "Weak similar effect on cell viability"
            elif correlation > -0.3:
                interaction = "Little to no relationship in cell viability effect"
            elif correlation > -0.5:
                interaction = "Weak opposing effect on cell viability"
            elif correlation > -0.7:
                interaction = "Moderate opposing effect on cell viability"
            else:
                interaction = "Strong opposing effect on cell viability"
        
        result = {
            "correlation": correlation,
            "p_value": p_value,
            "interaction": interaction
        }
        
        if "adjusted_p_value" in corr_data:
            result["adjusted_p_value"] = corr_data["adjusted_p_value"]
        
        return result
    
    def find_similar_genes(self, gene: str, top_n: int = 10, 
                          min_correlation: float = 0.5,
                          max_p_value: float = 0.05) -> List[Dict[str, Union[str, float]]]:
        """
        Find genes with similar effects on cell viability as the query gene.
        
        Parameters:
        -----------
        gene : str
            Query gene symbol
        top_n : int
            Number of top similar genes to return
        min_correlation : float
            Minimum correlation coefficient to consider
        max_p_value : float
            Maximum p-value to consider statistically significant
            
        Returns:
        --------
        list
            List of dictionaries with gene names and correlation statistics
        """
        if gene not in self.gene_to_idx:
            raise KeyError(f"Gene '{gene}' not found in dataset")
        
        idx = self.gene_to_idx[gene]
        similar_genes = []
        
        # Process based on storage format
        if self.format == "dense":
            # Get all correlations for this gene
            correlations = self.corr_matrix[idx, :]
            p_values = self.p_val_matrix[idx, :]
            
            # Create array of gene indices
            gene_indices = np.arange(self.num_genes)
            
            # Filter out the query gene itself, apply correlation and p-value filters
            mask = (gene_indices != idx) & (correlations >= min_correlation) & (p_values <= max_p_value)
            filtered_indices = gene_indices[mask]
            filtered_correlations = correlations[mask]
            filtered_p_values = p_values[mask]
            
            # Sort by correlation (descending)
            sorted_indices = np.argsort(filtered_correlations)[::-1]
            
            # Take top N
            top_indices = sorted_indices[:top_n]
            
            # Create result list
            for i in top_indices:
                gene_idx = filtered_indices[i]
                similar_genes.append({
                    "gene": self.gene_names[gene_idx],
                    "correlation": float(filtered_correlations[i]),
                    "p_value": float(filtered_p_values[i])
                })
        
        else:  # sparse format
            # For sparse format, we need to retrieve the entire row
            corr_data = self.h5_file['corr']
            p_val_data = self.h5_file['p_val']
            
            # Get row data for the query gene
            def get_csr_row(group, row):
                indptr = group['indptr'][:]
                indices = group['indices'][:]
                data = group['data'][:]
                
                row_indices = indices[indptr[row]:indptr[row+1]]
                row_data = data[indptr[row]:indptr[row+1]]
                
                return row_indices, row_data
            
            corr_indices, corr_values = get_csr_row(corr_data, idx)
            p_val_indices, p_val_values = get_csr_row(p_val_data, idx)
            
            # Create a dictionary for p-values
            p_val_dict = {int(i): float(v) for i, v in zip(p_val_indices, p_val_values)}
            
            # Filter and sort correlations
            candidates = []
            for i, v in zip(corr_indices, corr_values):
                i = int(i)
                if i == idx:  # Skip self
                    continue
                    
                corr = float(v)
                p_val = p_val_dict.get(i, 1.0)  # Default to 1.0 if not found
                
                if corr >= min_correlation and p_val <= max_p_value:
                    candidates.append((i, corr, p_val))
            
            # Sort by correlation (descending)
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Take top N
            for i, corr, p_val in candidates[:top_n]:
                similar_genes.append({
                    "gene": self.gene_names[i],
                    "correlation": corr,
                    "p_value": p_val
                })
        
        return similar_genes
    
    def __del__(self):
        """Clean up HDF5 file handle if using sparse format"""
        if hasattr(self, 'format') and self.format == "sparse" and hasattr(self, 'h5_file'):
            self.h5_file.close()


def compute_depmap24q2_gene_correlations(gene_a, gene_b, data_dir=None):
    """
    Robust cell viability correlation analysis between two genes using DepMap 24Q2 CERES data.
    
    This function computes Pearson correlation coefficients between gene knockout effects
    across 1,320 cancer cell lines, providing evidence-based insights into genetic 
    dependencies and cell viability relationships.
    
    Parameters:
    -----------
    gene_a : str
        First gene symbol for correlation analysis
    gene_b : str  
        Second gene symbol for correlation analysis
    data_dir : str, optional
        Path to DepMap 24Q2 preprocessed correlation data. If None, uses MEDEADB_PATH environment variable.
        
    Returns:
    --------
    tuple
        (correlation_coefficient, p_value, adjusted_p_value)
        Returns (None, None, None) if analysis fails
    """
    # Set default data_dir if not provided
    if data_dir is None:
        data_dir = os.path.join(_get_medeadb_path(), "depmap_24q2")
    
    def _log_insight(message, level="ANALYSIS"):
        """Structured logging for cell viability insights"""
        print(f"[DEPMAP] {level}: {message}", flush=True)
    
    def _validate_gene_symbols(gene_a, gene_b):
        """Validate and standardize gene symbols"""
        # Convert to uppercase for consistency
        gene_a_std = gene_a.upper().strip()
        gene_b_std = gene_b.upper().strip()
        
        # Basic validation
        if not gene_a_std or not gene_b_std:
            raise ValueError("Gene symbols cannot be empty")
        if gene_a_std == gene_b_std:
            _log_insight(f"Warning: Analyzing self-correlation for gene {gene_a_std}", "WARNING")
            
        return gene_a_std, gene_b_std
    
    def _interpret_correlation_strength(correlation):
        """Provide evidence-based interpretation of correlation strength"""
        abs_corr = abs(correlation)
        if abs_corr >= 0.8:
            return "very strong"
        elif abs_corr >= 0.6:
            return "strong" 
        elif abs_corr >= 0.4:
            return "moderate"
        elif abs_corr >= 0.2:
            return "weak"
        else:
            return "negligible"
    
    def _interpret_statistical_significance(p_value, adj_p_value=None):
        """Interpret statistical significance with multiple testing context"""
        if adj_p_value is not None and adj_p_value <= 0.001:
            return "highly significant (survives multiple testing correction)"
        elif adj_p_value is not None and adj_p_value <= 0.05:
            return "significant (survives multiple testing correction)"
        elif p_value <= 0.001:
            return "highly significant (uncorrected)"
        elif p_value <= 0.05:
            return "significant (uncorrected)"
        elif p_value <= 0.1:
            return "marginally significant"
        else:
            return "not statistically significant"
    
    def _generate_biological_interpretation(correlation, p_value, gene_a, gene_b, adj_p_value=None):
        """Generate evidence-based biological interpretation"""
        strength = _interpret_correlation_strength(correlation)
        significance = _interpret_statistical_significance(p_value, adj_p_value)
        
        if p_value > 0.05:
            return f"No reliable evidence for correlated cell viability effects between {gene_a} and {gene_b}"
        
        direction = "similar" if correlation > 0 else "opposing"
        
        # Detailed biological context
        if abs(correlation) >= 0.6 and p_value <= 0.001:
            confidence = "high confidence"
            evidence = "strong empirical evidence"
        elif abs(correlation) >= 0.4 and p_value <= 0.05:
            confidence = "moderate confidence" 
            evidence = "substantial evidence"
        else:
            confidence = "low confidence"
            evidence = "limited evidence"
            
        interpretation = (f"{confidence.capitalize()} of {direction} cell viability effects "
                         f"({strength} correlation, {significance}). "
                         f"This suggests {evidence} for ")
        
        if correlation > 0:
            interpretation += f"co-dependency or shared pathway involvement between {gene_a} and {gene_b}"
        else:
            interpretation += f"compensatory or antagonistic relationship between {gene_a} and {gene_b}"
            
        return interpretation
    
    try:
        # Input validation and standardization
        _log_insight(f"Initiating cell viability correlation analysis: {gene_a} ↔ {gene_b}")
        gene_a_std, gene_b_std = _validate_gene_symbols(gene_a, gene_b)
        
        # Data source validation
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"DepMap data directory not accessible: {data_dir}")
        
        _log_insight(f"Gene symbols standardized: {gene_a_std}, {gene_b_std}")
        _log_insight(f"Loading DepMap 24Q2 correlation matrix from: {data_dir}")
        
        # Initialize correlation lookup with error handling
        try:
            lookup = GeneCorrelationLookup(data_dir)
            _log_insight(f"Successfully loaded correlation data for {lookup.num_genes:,} genes")
        except Exception as e:
            _log_insight(f"Failed to initialize DepMap correlation data: {str(e)}", "ERROR")
            raise
        
        # Gene availability validation
        missing_genes = []
        if gene_a_std not in lookup.gene_to_idx:
            missing_genes.append(gene_a_std)
        if gene_b_std not in lookup.gene_to_idx:
            missing_genes.append(gene_b_std)
            
        if missing_genes:
            _log_insight(f"Genes not found in DepMap dataset: {', '.join(missing_genes)}", "ERROR")
            _log_insight(f"Available genes: {lookup.num_genes:,} total in DepMap 24Q2", "INFO")
            raise KeyError(f"Gene(s) not available in DepMap correlation matrix: {', '.join(missing_genes)}")
        
        _log_insight(f"Gene pair validation successful - both genes found in dataset")
        
        # Perform correlation analysis
        _log_insight(f"Computing CERES-based correlation across 1,320 cancer cell lines")
        result = lookup.get_cell_viability_effect(gene_a_std, gene_b_std)
        
        correlation = result['correlation']
        p_value = result['p_value'] 
        adj_p_value = result.get('adjusted_p_value')
        
        # Data quality validation
        if not (-1 <= correlation <= 1):
            _log_insight(f"Warning: Correlation value outside expected range: {correlation}", "WARNING")
        if p_value < 0 or p_value > 1:
            _log_insight(f"Warning: P-value outside expected range: {p_value}", "WARNING")
            
        # Statistical analysis summary
        _log_insight(f"Correlation coefficient: {correlation:.4f}")
        _log_insight(f"Statistical significance: p = {p_value:.2e}")
        if adj_p_value is not None:
            _log_insight(f"Multiple testing correction: adj_p = {adj_p_value:.2e}")
        
        # Evidence-based interpretation
        biological_interpretation = _generate_biological_interpretation(
            correlation, p_value, gene_a_std, gene_b_std, adj_p_value
        )
        _log_insight(f"Biological interpretation: {biological_interpretation}")
        
        # Clinical relevance context
        if abs(correlation) >= 0.5 and p_value <= 0.01:
            _log_insight(f"Clinical relevance: Strong correlation suggests potential for combination "
                        f"therapy or synthetic lethality screening", "INSIGHT")
        elif abs(correlation) >= 0.3 and p_value <= 0.05:
            _log_insight(f"Research priority: Moderate correlation warrants further investigation "
                        f"in relevant cancer contexts", "INSIGHT")
        
        # Analysis summary
        _log_insight(f"Analysis completed successfully - correlation: {correlation:.4f}, "
                    f"significance: {_interpret_statistical_significance(p_value, adj_p_value)}")
        
        return correlation, p_value, adj_p_value
        
    except KeyError as e:
        _log_insight(f"Gene availability error: {str(e)}", "ERROR")
        _log_insight(f"Recommendation: Verify gene symbols or check DepMap gene coverage", "INFO")
        return None, None, None
        
    except FileNotFoundError as e:
        _log_insight(f"Data access error: {str(e)}", "ERROR")
        _log_insight(f"Recommendation: Verify DepMap data directory path and permissions", "INFO")
        return None, None, None
        
    except ValueError as e:
        _log_insight(f"Input validation error: {str(e)}", "ERROR")
        _log_insight(f"Recommendation: Check gene symbol format and validity", "INFO")
        return None, None, None
        
    except Exception as e:
        _log_insight(f"Unexpected analysis error: {str(e)}", "ERROR")
        _log_insight(f"System context: DepMap correlation analysis framework", "DEBUG")
        return None, None, None

#### Enrichr Tools ####
@dataclass
class InteractionResult:
    """Data class to store genetic interaction prediction results."""
    gene_pair: Tuple[str, str]
    interaction_score: float
    confidence_level: str
    evidence_pathways: List[str]
    predicted_interaction_type: str
    supporting_data: Dict[str, any]


class BaseEnrichrInteractionTool(ABC):
    """Base class for individual Enrichr library interaction tools."""
    
    ENRICHR_URL = "https://maayanlab.cloud/Enrichr/addList"
    ENRICHMENT_URL = "https://maayanlab.cloud/Enrichr/enrich"
    
    def __init__(self, library_name: str, tool_name: str, top_results: int = 5):
        self.library_name = library_name
        self.tool_name = tool_name
        self.top_results = top_results
        self.gene_cache = {}
        self.request_delay = 1.0
        self.max_retries = 3
        
    def _log(self, message: str, level: str = "INSIGHT"):
        """Log insights for reasoning analysis."""
        prefix = f"[{self.tool_name}]"
        print(f"{prefix} {level}: {message}", flush=True)
        
    def _rate_limited_request(self, request_func, *args, **kwargs):
        """Execute request with rate limiting and exponential backoff."""
        for attempt in range(self.max_retries):
            try:
                time.sleep(self.request_delay)
                response = request_func(*args, **kwargs)
                response.raise_for_status()
                return response
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    wait_time = (2 ** attempt) * self.request_delay
                    if attempt == self.max_retries - 1:
                        self._log(f"API rate limit exceeded - analysis may be incomplete", "WARNING")
                        raise Exception(f"Rate limit exceeded after {self.max_retries} attempts")
                    time.sleep(wait_time)
                else:
                    raise e
            except requests.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise Exception(f"Request failed after {self.max_retries} attempts: {e}")
                time.sleep(2 ** attempt)
    
    def get_official_gene_name(self, gene_name: str) -> str:
        """Get official gene symbol with insight logging."""
        if gene_name in self.gene_cache:
            return self.gene_cache[gene_name]
            
        encoded_gene_name = urllib.parse.quote(gene_name)
        url = f"https://mygene.info/v3/query?q={encoded_gene_name}&fields=symbol,alias&species=human"
        
        try:
            response = self._rate_limited_request(requests.get, url, timeout=10)
            data = response.json()
            hits = data.get("hits", [])
            
            if not hits:
                self._log(f"Gene '{gene_name}' not found in database - using original name", "WARNING")
                self.gene_cache[gene_name] = gene_name
                return gene_name

            # Find exact match
            for hit in hits:
                symbol = hit.get("symbol", "")
                if symbol.upper() == gene_name.upper():
                    if symbol != gene_name:
                        self._log(f"Gene name standardized: {gene_name} → {symbol}")
                    self.gene_cache[gene_name] = symbol
                    return symbol
                aliases = hit.get("alias", [])
                if any(gene_name.upper() == alias.upper() for alias in aliases):
                    self._log(f"Gene alias resolved: {gene_name} → {symbol}")
                    self.gene_cache[gene_name] = symbol
                    return symbol

            # Use top hit
            top_hit = hits[0]
            symbol = top_hit.get("symbol", gene_name)
            if symbol != gene_name:
                self._log(f"Gene name corrected: {gene_name} → {symbol}")
            self.gene_cache[gene_name] = symbol
            return symbol
            
        except Exception as e:
            self._log(f"Gene name resolution failed for '{gene_name}' - proceeding with original", "WARNING")
            self.gene_cache[gene_name] = gene_name
            return gene_name

    def submit_gene_list(self, genes: List[str]) -> str:
        """Submit gene list to Enrichr."""
        gene_list_str = "\n".join(genes)
        payload = {
            "list": (None, gene_list_str),
            "description": (None, f"Gene pair analysis: {' + '.join(genes)}")
        }
        
        try:
            response = self._rate_limited_request(requests.post, self.ENRICHR_URL, files=payload, timeout=30)
            user_list_id = response.json()["userListId"]
            return user_list_id
        except Exception as e:
            self._log(f"Failed to submit genes to Enrichr: {e}", "ERROR")
            raise Exception(f"Error submitting gene list to Enrichr: {e}")

    def get_enrichment_results(self, user_list_id: str) -> List:
        """Fetch enrichment results."""
        query_string = f"?userListId={user_list_id}&backgroundType={self.library_name}"
        
        try:
            response = self._rate_limited_request(requests.get, self.ENRICHMENT_URL + query_string, timeout=30)
            results = response.json()
            library_results = results.get(self.library_name, [])[:self.top_results]
            return library_results
        except Exception as e:
            self._log(f"Failed to retrieve enrichment data: {e}", "ERROR")
            raise Exception(f"Error fetching enrichment results: {e}")

    @abstractmethod
    def analyze_gene_pair(self, gene1: str, gene2: str) -> Tuple[str, str, List[str], List[str]]:
        """
        Analyze gene pair interaction and return standardized 4-element output.
        Returns: (summary, confidence, top_mechanisms, key_pathways)
        """
        pass

    def _extract_shared_terms(self, gene1: str, gene2: str, enrichment_results: List) -> Tuple[List[str], float]:
        """Extract shared terms and calculate interaction score with insights."""
        shared_terms = []
        interaction_score = 0.0
        high_significance_terms = []
        
        for result in enrichment_results:
            if len(result) < 6:
                continue
                
            term_name = result[1]
            p_value = result[2]
            associated_genes = result[5]
            
            if gene1 in associated_genes and gene2 in associated_genes:
                shared_terms.append(term_name)
                score_contribution = -np.log10(max(p_value, 1e-10))
                interaction_score += score_contribution
                
                # Log highly significant findings
                if p_value < 0.001:
                    high_significance_terms.append((term_name, p_value))
        
        if high_significance_terms:
            top_term, top_p = high_significance_terms[0]
            self._log(f"Strong evidence found: '{top_term}' (p={top_p:.2e})")
        
        if shared_terms:
            self._log(f"Interaction detected: {len(shared_terms)} shared terms, score={interaction_score:.2f}")
        else:
            self._log("No shared terms detected between gene pair")
        
        return shared_terms, interaction_score

    def _calculate_confidence(self, shared_count: int, interaction_score: float) -> str:
        """Calculate confidence level with reasoning insight."""
        if shared_count == 0:
            return "none"
        
        # Normalized confidence score
        confidence_score = min(1.0, (shared_count * interaction_score) / 50.0)
        
        if confidence_score >= 0.6:
            confidence = "high"
        elif confidence_score >= 0.3:
            confidence = "medium"
        elif confidence_score >= 0.1:
            confidence = "low"
        else:
            confidence = "very_low"
        
        # Log reasoning for confidence level
        factors = []
        if shared_count >= 3:
            factors.append(f"multiple shared terms ({shared_count})")
        if interaction_score >= 10:
            factors.append("high statistical significance")
        if shared_count == 1:
            factors.append("single shared term")
        
        if factors:
            self._log(f"Confidence '{confidence}' based on: {', '.join(factors)}")
        
        return confidence


class WikiPathwaysInteractionTool(BaseEnrichrInteractionTool):
    """Individual tool for WikiPathways-based genetic interaction analysis."""
    
    def __init__(self):
        super().__init__("WikiPathways_2024_Human", "PATHWAY", top_results=5)
        
    def analyze_gene_pair(self, gene1: str, gene2: str) -> Tuple[str, str, List[str], List[str]]:
        """
        Analyze gene pair using WikiPathways data.
        Returns: (summary, confidence, top_mechanisms, key_pathways)
        """
        self._log(f"Analyzing pathway interactions for {gene1} + {gene2}")
        
        # Resolve gene names
        official_gene1 = self.get_official_gene_name(gene1)
        official_gene2 = self.get_official_gene_name(gene2)
        
        try:
            # Submit genes and get enrichment
            user_list_id = self.submit_gene_list([official_gene1, official_gene2])
            enrichment_results = self.get_enrichment_results(user_list_id)
            
            if not enrichment_results:
                self._log("No enriched pathways found in WikiPathways database")
                return ("No pathway interactions found", "none", [], [])
            
            # Extract shared pathways
            shared_pathways, interaction_score = self._extract_shared_terms(official_gene1, official_gene2, enrichment_results)
            
            if not shared_pathways:
                self._log("Genes do not co-occur in same pathways")
                return ("Genes do not share significant pathways", "none", [], [])
            
            # Analyze pathway types for biological insights
            pathway_text = " ".join(shared_pathways).lower()
            mechanisms = []
            pathway_insights = []
            
            if any(term in pathway_text for term in ["signaling", "signal", "cascade"]):
                mechanisms.append("signaling_pathway")
                pathway_insights.append("signal transduction")
            if any(term in pathway_text for term in ["metabolic", "metabolism", "synthesis"]):
                mechanisms.append("metabolic_pathway")
                pathway_insights.append("metabolic regulation")
            if any(term in pathway_text for term in ["dna repair", "cell cycle", "apoptosis"]):
                mechanisms.append("regulatory_pathway")
                pathway_insights.append("cell cycle control")
            if any(term in pathway_text for term in ["transport", "trafficking"]):
                mechanisms.append("transport_pathway")
                pathway_insights.append("cellular transport")
            
            if not mechanisms:
                mechanisms = ["general_pathway"]
                pathway_insights.append("general pathway interaction")
            
            # Log biological insights
            if pathway_insights:
                self._log(f"Biological mechanisms identified: {', '.join(pathway_insights)}")
            
            # Calculate confidence
            confidence = self._calculate_confidence(len(shared_pathways), interaction_score)
            
            # Create summary with biological context
            summary = f"Genes interact through {len(shared_pathways)} shared pathway(s) with {confidence} confidence"
            
            # Limit pathways to top 5
            key_pathways = shared_pathways[:5]
            
            self._log(f"Pathway analysis complete: {confidence} confidence, {len(mechanisms)} mechanism types")
            
            return (summary, confidence, mechanisms, key_pathways)
            
        except Exception as e:
            self._log(f"Pathway analysis failed: {str(e)}", "ERROR")
            return ("Pathway analysis failed", "none", [], [])


class ReactomeInteractionTool(BaseEnrichrInteractionTool):
    """Individual tool for Reactome-based genetic interaction analysis."""
    
    def __init__(self):
        super().__init__("Reactome_Pathways_2024", "REACTOME", top_results=5)
        
    def analyze_gene_pair(self, gene1: str, gene2: str) -> Tuple[str, str, List[str], List[str]]:
        """
        Analyze gene pair using Reactome data.
        Returns: (summary, confidence, top_mechanisms, key_pathways)
        """
        self._log(f"Analyzing biochemical interactions for {gene1} + {gene2}")
        
        official_gene1 = self.get_official_gene_name(gene1)
        official_gene2 = self.get_official_gene_name(gene2)
        
        try:
            user_list_id = self.submit_gene_list([official_gene1, official_gene2])
            enrichment_results = self.get_enrichment_results(user_list_id)
            
            if not enrichment_results:
                self._log("No biochemical reactions found in Reactome database")
                return ("No biochemical interactions found", "none", [], [])
            
            shared_reactions, interaction_score = self._extract_shared_terms(official_gene1, official_gene2, enrichment_results)
            
            if not shared_reactions:
                self._log("Genes do not participate in same biochemical reactions")
                return ("Genes do not share biochemical reactions", "none", [], [])
            
            # Analyze reaction types for molecular insights
            mechanisms = []
            molecular_insights = []
            reaction_text = " ".join(shared_reactions).lower()
            
            if "phosphorylation" in reaction_text:
                mechanisms.append("phosphorylation_regulation")
                molecular_insights.append("phosphorylation signaling")
            if "complex" in reaction_text:
                mechanisms.append("protein_complex_formation")
                molecular_insights.append("protein complex assembly")
            if "binding" in reaction_text:
                mechanisms.append("direct_protein_binding")
                molecular_insights.append("direct molecular binding")
            if "transport" in reaction_text:
                mechanisms.append("transport_mechanism")
                molecular_insights.append("molecular transport")
            if "degradation" in reaction_text:
                mechanisms.append("protein_degradation")
                molecular_insights.append("protein degradation")
            
            if not mechanisms:
                mechanisms = ["biochemical_interaction"]
                molecular_insights.append("general biochemical interaction")
            
            # Log molecular insights
            if molecular_insights:
                self._log(f"Molecular mechanisms identified: {', '.join(molecular_insights)}")
            
            confidence = self._calculate_confidence(len(shared_reactions), interaction_score)
            summary = f"Genes interact through {len(shared_reactions)} shared biochemical reaction(s) with {confidence} confidence"
            key_pathways = shared_reactions[:5]
            
            self._log(f"Reactome analysis complete: {confidence} confidence, molecular interaction detected")
            
            return (summary, confidence, mechanisms, key_pathways)
            
        except Exception as e:
            self._log(f"Reactome analysis failed: {str(e)}", "ERROR")
            return ("Reactome analysis failed", "none", [], [])


class MSigDBHallmarkInteractionTool(BaseEnrichrInteractionTool):
    """Individual tool for MSigDB Hallmark-based genetic interaction analysis."""
    
    def __init__(self):
        super().__init__("MSigDB_Hallmark_2020", "HALLMARK", top_results=5)
        
    def analyze_gene_pair(self, gene1: str, gene2: str) -> Tuple[str, str, List[str], List[str]]:
        """
        Analyze gene pair using MSigDB Hallmarks.
        Returns: (summary, confidence, top_mechanisms, key_pathways)
        """
        self._log(f"Analyzing cancer hallmark processes for {gene1} + {gene2}")
        
        official_gene1 = self.get_official_gene_name(gene1)
        official_gene2 = self.get_official_gene_name(gene2)
        
        try:
            user_list_id = self.submit_gene_list([official_gene1, official_gene2])
            enrichment_results = self.get_enrichment_results(user_list_id)
            
            if not enrichment_results:
                self._log("No cancer hallmark processes identified")
                return ("No hallmark processes found", "none", [], [])
            
            shared_hallmarks, interaction_score = self._extract_shared_terms(official_gene1, official_gene2, enrichment_results)
            
            if not shared_hallmarks:
                self._log("Genes do not co-regulate cancer hallmark processes")
                return ("Genes do not share cancer hallmark processes", "none", [], [])
            
            # Analyze hallmark types for cancer biology insights
            mechanisms = []
            cancer_insights = []
            hallmark_text = " ".join(shared_hallmarks).lower()
            
            if "apoptosis" in hallmark_text:
                mechanisms.append("apoptotic_regulation")
                cancer_insights.append("cell death regulation")
            if any(term in hallmark_text for term in ["proliferation", "cell_cycle"]):
                mechanisms.append("proliferative_control")
                cancer_insights.append("growth control")
            if "metabolism" in hallmark_text:
                mechanisms.append("metabolic_reprogramming")
                cancer_insights.append("metabolic rewiring")
            if any(term in hallmark_text for term in ["immune", "inflammatory"]):
                mechanisms.append("immune_response")
                cancer_insights.append("immune system interaction")
            if "dna_repair" in hallmark_text:
                mechanisms.append("dna_repair_mechanism")
                cancer_insights.append("genome stability")
            if "angiogenesis" in hallmark_text:
                mechanisms.append("angiogenesis_regulation")
                cancer_insights.append("blood vessel formation")
            
            if not mechanisms:
                mechanisms = ["hallmark_process"]
                cancer_insights.append("general cancer process")
            
            # Log cancer biology insights
            if cancer_insights:
                self._log(f"Cancer biology roles identified: {', '.join(cancer_insights)}")
            
            # Hallmarks get higher confidence weighting due to curated nature
            confidence = self._calculate_confidence(len(shared_hallmarks), interaction_score * 1.5)
            summary = f"Genes co-regulate {len(shared_hallmarks)} cancer hallmark process(es) with {confidence} confidence"
            key_pathways = shared_hallmarks[:5]
            
            self._log(f"Hallmark analysis complete: {confidence} confidence, cancer relevance established")
            
            return (summary, confidence, mechanisms, key_pathways)
            
        except Exception as e:
            self._log(f"Hallmark analysis failed: {str(e)}", "ERROR")
            return ("Hallmark analysis failed", "none", [], [])


class GOFunctionInteractionTool(BaseEnrichrInteractionTool):
    """Individual tool for GO Molecular Function-based genetic interaction analysis."""
    
    def __init__(self):
        super().__init__("GO_Molecular_Function_2023", "GO_FUNCTION", top_results=5)
        
    def analyze_gene_pair(self, gene1: str, gene2: str) -> Tuple[str, str, List[str], List[str]]:
        """
        Analyze gene pair using GO Molecular Functions.
        Returns: (summary, confidence, top_mechanisms, key_pathways)
        """
        self._log(f"Analyzing molecular functions for {gene1} + {gene2}")
        
        official_gene1 = self.get_official_gene_name(gene1)
        official_gene2 = self.get_official_gene_name(gene2)
        
        try:
            user_list_id = self.submit_gene_list([official_gene1, official_gene2])
            enrichment_results = self.get_enrichment_results(user_list_id)
            
            if not enrichment_results:
                self._log("No shared molecular functions identified")
                return ("No shared molecular functions found", "none", [], [])
            
            shared_functions, interaction_score = self._extract_shared_terms(official_gene1, official_gene2, enrichment_results)
            
            if not shared_functions:
                self._log("Genes have distinct molecular function profiles")
                return ("Genes do not share molecular functions", "none", [], [])
            
            # Analyze function types for biochemical insights
            mechanisms = []
            functional_insights = []
            function_text = " ".join(shared_functions).lower()
            
            if "kinase" in function_text:
                mechanisms.append("kinase_activity")
                functional_insights.append("protein phosphorylation")
            if "phosphatase" in function_text:
                mechanisms.append("phosphatase_activity")
                functional_insights.append("protein dephosphorylation")
            if "binding" in function_text:
                mechanisms.append("protein_binding")
                functional_insights.append("molecular binding")
            if "transcription" in function_text:
                mechanisms.append("transcriptional_regulation")
                functional_insights.append("gene expression control")
            if "enzyme" in function_text:
                mechanisms.append("enzymatic_activity")
                functional_insights.append("catalytic activity")
            if "receptor" in function_text:
                mechanisms.append("receptor_activity")
                functional_insights.append("signal reception")
            
            if not mechanisms:
                mechanisms = ["molecular_function"]
                functional_insights.append("shared molecular activity")
            
            # Log functional insights
            if functional_insights:
                self._log(f"Functional activities identified: {', '.join(functional_insights)}")
            
            confidence = self._calculate_confidence(len(shared_functions), interaction_score)
            summary = f"Genes share {len(shared_functions)} molecular function(s) with {confidence} confidence"
            key_pathways = shared_functions[:5]
            
            self._log(f"Function analysis complete: {confidence} confidence, functional similarity detected")
            
            return (summary, confidence, mechanisms, key_pathways)
            
        except Exception as e:
            self._log(f"Function analysis failed: {str(e)}", "ERROR")
            return ("Function analysis failed", "none", [], [])


class GOProcessInteractionTool(BaseEnrichrInteractionTool):
    """Individual tool for GO Biological Process-based genetic interaction analysis."""
    
    def __init__(self):
        super().__init__("GO_Biological_Process_2023", "GO_PROCESS", top_results=5)
        
    def analyze_gene_pair(self, gene1: str, gene2: str) -> Tuple[str, str, List[str], List[str]]:
        """
        Analyze gene pair using GO Biological Processes.
        Returns: (summary, confidence, top_mechanisms, key_pathways)
        """
        self._log(f"Analyzing biological processes for {gene1} + {gene2}")
        
        official_gene1 = self.get_official_gene_name(gene1)
        official_gene2 = self.get_official_gene_name(gene2)
        
        try:
            user_list_id = self.submit_gene_list([official_gene1, official_gene2])
            enrichment_results = self.get_enrichment_results(user_list_id)
            
            if not enrichment_results:
                self._log("No shared biological processes identified")
                return ("No shared biological processes found", "none", [], [])
            
            shared_processes, interaction_score = self._extract_shared_terms(official_gene1, official_gene2, enrichment_results)
            
            if not shared_processes:
                self._log("Genes participate in distinct biological processes")
                return ("Genes do not share biological processes", "none", [], [])
            
            # Analyze process types for biological insights
            mechanisms = []
            process_insights = []
            process_text = " ".join(shared_processes).lower()
            
            if any(term in process_text for term in ["cell cycle", "mitosis", "division"]):
                mechanisms.append("cell_cycle_regulation")
                process_insights.append("cell division control")
            if any(term in process_text for term in ["apoptosis", "cell death"]):
                mechanisms.append("apoptotic_process")
                process_insights.append("programmed cell death")
            if "signal transduction" in process_text:
                mechanisms.append("signal_transduction")
                process_insights.append("cellular signaling")
            if any(term in process_text for term in ["dna repair", "dna damage"]):
                mechanisms.append("dna_repair_process")
                process_insights.append("genome maintenance")
            if "metabolic" in process_text:
                mechanisms.append("metabolic_process")
                process_insights.append("metabolic regulation")
            if "development" in process_text:
                mechanisms.append("developmental_process")
                process_insights.append("developmental biology")
            
            if not mechanisms:
                mechanisms = ["biological_process"]
                process_insights.append("shared cellular process")
            
            # Log process insights
            if process_insights:
                self._log(f"Biological processes identified: {', '.join(process_insights)}")
            
            confidence = self._calculate_confidence(len(shared_processes), interaction_score)
            summary = f"Genes participate in {len(shared_processes)} shared biological process(es) with {confidence} confidence"
            key_pathways = shared_processes[:5]
            
            self._log(f"Process analysis complete: {confidence} confidence, process co-participation detected")
            
            return (summary, confidence, mechanisms, key_pathways)
            
        except Exception as e:
            self._log(f"Process analysis failed: {str(e)}", "ERROR")
            return ("Process analysis failed", "none", [], [])


# Convenience functions for easy tool access
def analyze_pathway_interaction(gene1: str, gene2: str) -> Tuple[str, str, List[str], List[str]]:
    """Analyze gene pair using WikiPathways."""
    tool = WikiPathwaysInteractionTool()
    return tool.analyze_gene_pair(gene1, gene2)


def analyze_reactome_interaction(gene1: str, gene2: str) -> Tuple[str, str, List[str], List[str]]:
    """Analyze gene pair using Reactome."""
    tool = ReactomeInteractionTool()
    return tool.analyze_gene_pair(gene1, gene2)


def analyze_hallmark_interaction(gene1: str, gene2: str) -> Tuple[str, str, List[str], List[str]]:
    """Analyze gene pair using MSigDB Hallmarks."""
    tool = MSigDBHallmarkInteractionTool()
    return tool.analyze_gene_pair(gene1, gene2)


def analyze_function_interaction(gene1: str, gene2: str) -> Tuple[str, str, List[str], List[str]]:
    """Analyze gene pair using GO Molecular Functions."""
    tool = GOFunctionInteractionTool()
    return tool.analyze_gene_pair(gene1, gene2)


def analyze_process_interaction(gene1: str, gene2: str) -> Tuple[str, str, List[str], List[str]]:
    """Analyze gene pair using GO Biological Processes."""
    tool = GOProcessInteractionTool()
    return tool.analyze_gene_pair(gene1, gene2)


def analyze_comprehensive_interaction(gene1: str, gene2: str) -> Dict[str, Tuple[str, str, List[str], List[str]]]:
    """
    Run all interaction analysis tools on a gene pair.
    Returns a dictionary with results from each tool.
    """
    print(f"\n=== COMPREHENSIVE INTERACTION ANALYSIS ===")
    print(f"Analyzing gene pair: ({gene1}, {gene2})")
    print("=" * 50)
    
    tools = {
        'pathway': WikiPathwaysInteractionTool(),
        'reactome': ReactomeInteractionTool(),
        'hallmark': MSigDBHallmarkInteractionTool(),
        'function': GOFunctionInteractionTool(),
        'process': GOProcessInteractionTool()
    }
    
    results = {}
    
    for tool_name, tool in tools.items():
        try:
            print(f"\n--- {tool_name.upper()} ANALYSIS ---")
            result = tool.analyze_gene_pair(gene1, gene2)
            results[tool_name] = result
            
            summary, confidence, mechanisms, pathways = result
            print(f"Summary: {summary}")
            print(f"Confidence: {confidence}")
            print(f"Mechanisms: {mechanisms}")
            print(f"Key Evidence: {pathways[:3] if pathways else 'None'}")
            
        except Exception as e:
            print(f"ERROR in {tool_name}: {e}")
            results[tool_name] = ("Analysis failed", "none", [], [])
    
    return results


# Legacy compatibility function
def enrichr_api(genes: list, libs: list = None) -> dict:
    """
    Legacy function for backward compatibility.
    """
    if len(genes) < 2:
        raise ValueError("At least 2 genes required for gene pair analysis")
    
    gene1, gene2 = genes[0], genes[1]
    
    if libs is None:
        libs = ["WikiPathways_2024_Human", "Reactome_Pathways_2024", "MSigDB_Hallmark_2020"]
    
    tool_mapping = {
        "WikiPathways_2024_Human": analyze_pathway_interaction,
        "Reactome_Pathways_2024": analyze_reactome_interaction,
        "MSigDB_Hallmark_2020": analyze_hallmark_interaction,
        "GO_Molecular_Function_2023": analyze_function_interaction,
        "GO_Biological_Process_2023": analyze_process_interaction
    }
    
    results = {}
    
    for lib in libs:
        if lib in tool_mapping:
            try:
                summary, confidence, mechanisms, pathways = tool_mapping[lib](gene1, gene2)
                results[lib] = {
                    "summary": summary,
                    "confidence": confidence,
                    "mechanisms": mechanisms,
                    "pathways": pathways
                }
            except Exception as e:
                print(f"Error with {lib}: {e}")
                results[lib] = {
                    "summary": "Analysis failed",
                    "confidence": "none",
                    "mechanisms": [],
                    "pathways": []
                }
    
    return results

#### Human Protein Atlas (HPA) Tools ####
HPA_SEARCH_API = "https://www.proteinatlas.org/api/search_download.php"
HPA_BASE = "https://www.proteinatlas.org"
HPA_JSON_API_TEMPLATE = "https://www.proteinatlas.org/{ensembl_id}.json"
HPA_XML_API_TEMPLATE = "https://www.proteinatlas.org/{ensembl_id}.xml"

# --- Base Tool Classes ---

class HPASearchApiTool:
    """
    Base class for interacting with HPA's search_download.php API.
    Uses HPA's search and download API to get protein expression data.
    """
    def __init__(self, tool_config):
        self.timeout = 30
        self.base_url = HPA_SEARCH_API

    def _make_api_request(self, search_term: str, columns: str, format_type: str = "json") -> Dict[str, Any]:
        """Make HPA API request with improved error handling"""
        params = {
            "search": search_term,
            "format": format_type,
            "columns": columns,
            "compress": "no"
        }
        
        try:
            resp = requests.get(self.base_url, params=params, timeout=self.timeout)
            if resp.status_code == 404:
                return {"error": f"No data found for gene '{search_term}'"}
            if resp.status_code != 200:
                return {"error": f"HPA API request failed, HTTP {resp.status_code}", "detail": resp.text}
            
            if format_type == "json":
                data = resp.json()
                # Ensure we always return a list for consistency
                if not isinstance(data, list):
                    return {"error": "API did not return expected list format"}
                return data
            else:
                return {"tsv_data": resp.text}
                
        except requests.RequestException as e:
            return {"error": f"HPA API request failed: {str(e)}"}
        except ValueError as e:
            return {"error": f"Failed to parse HPA response data: {str(e)}", "content": resp.text}


class HPAJsonApiTool:
    """
    Base class for interacting with HPA's /{ensembl_id}.json API.
    More efficient for getting comprehensive gene data.
    """
    def __init__(self, tool_config):
        self.timeout = 30
        self.base_url_template = HPA_JSON_API_TEMPLATE

    def _make_api_request(self, ensembl_id: str) -> Dict[str, Any]:
        """Make HPA JSON API request for a specific gene"""
        url = self.base_url_template.format(ensembl_id=ensembl_id)
        try:
            resp = requests.get(url, timeout=self.timeout)
            if resp.status_code == 404:
                return {"error": f"No data found for Ensembl ID '{ensembl_id}'"}
            if resp.status_code != 200:
                return {"error": f"HPA JSON API request failed, HTTP {resp.status_code}", "detail": resp.text}
            
            return resp.json()
                
        except requests.RequestException as e:
            return {"error": f"HPA JSON API request failed: {str(e)}"}
        except ValueError as e:
            return {"error": f"Failed to parse HPA JSON response: {str(e)}", "content": resp.text}


class HPAXmlApiTool:
    """
    Base class for interacting with HPA's /{ensembl_id}.xml API.
    Optimized for comprehensive XML data extraction.
    """
    def __init__(self, tool_config):
        self.timeout = 45
        self.base_url_template = HPA_XML_API_TEMPLATE
    
    def _make_api_request(self, ensembl_id: str) -> ET.Element:
        """Make HPA XML API request for a specific gene"""
        url = self.base_url_template.format(ensembl_id=ensembl_id)
        try:
            resp = requests.get(url, timeout=self.timeout)
            if resp.status_code == 404:
                raise Exception(f"No XML data found for Ensembl ID '{ensembl_id}'")
            if resp.status_code != 200:
                raise Exception(f"HPA XML API request failed, HTTP {resp.status_code}")
            
            return ET.fromstring(resp.content)
        except requests.RequestException as e:
            raise Exception(f"HPA XML API request failed: {str(e)}")
        except ET.ParseError as e:
            raise Exception(f"Failed to parse HPA XML response: {str(e)}")


# --- New Enhanced Tools Based on Your Optimization Plan ---

class HPAGetRnaExpressionBySourceTool(HPASearchApiTool):
    """
    Get RNA expression for a gene from specific biological sources using optimized columns parameter.
    This tool directly leverages the comprehensive columns table for efficient queries.
    """
    def __init__(self, tool_config):
        super().__init__(tool_config)
        # Use correct HPA API column identifiers
        self.source_column_mappings = {
            "tissue": "rnatsm",      # RNA tissue specific nTPM
            "blood": "rnablm",       # RNA blood lineage specific nTPM  
            "brain": "rnabrm",       # RNA brain region specific nTPM
            "single_cell": "rnascm"  # RNA single cell type specific nTPM
        }
        
        # Map expected API response field names for each source type
        self.api_response_fields = {
            "tissue": "RNA tissue specific nTPM",
            "blood": "RNA blood lineage specific nTPM", 
            "brain": "RNA brain region specific nTPM",
            "single_cell": "RNA single cell type specific nTPM"
        }
        
        # Map source names to expected keys in API response
        self.source_name_mappings = {
            "tissue": {
                "adipose_tissue": ["adipose tissue", "fat"],
                "adrenal_gland": ["adrenal gland", "adrenal"],
                "appendix": ["appendix"],
                "bone_marrow": ["bone marrow"],
                "brain": ["brain", "cerebral cortex"],
                "breast": ["breast"],
                "bronchus": ["bronchus"],
                "cerebellum": ["cerebellum"],
                "cerebral_cortex": ["cerebral cortex", "brain"],
                "cervix": ["cervix"],
                "choroid_plexus": ["choroid plexus"],
                "colon": ["colon"],
                "duodenum": ["duodenum"],
                "endometrium": ["endometrium"],
                "epididymis": ["epididymis"],
                "esophagus": ["esophagus"],
                "fallopian_tube": ["fallopian tube"],
                "gallbladder": ["gallbladder"],
                "heart_muscle": ["heart muscle", "heart"],
                "hippocampal_formation": ["hippocampus", "hippocampal formation"],
                "hypothalamus": ["hypothalamus"],
                "kidney": ["kidney"],
                "liver": ["liver"],
                "lung": ["lung"],
                "lymph_node": ["lymph node"],
                "nasopharynx": ["nasopharynx"],
                "oral_mucosa": ["oral mucosa"],
                "ovary": ["ovary"],
                "pancreas": ["pancreas"],
                "parathyroid_gland": ["parathyroid gland"],
                "pituitary_gland": ["pituitary gland"],
                "placenta": ["placenta"],
                "prostate": ["prostate"],
                "rectum": ["rectum"],
                "retina": ["retina"],
                "salivary_gland": ["salivary gland"],
                "seminal_vesicle": ["seminal vesicle"],
                "skeletal_muscle": ["skeletal muscle"],
                "skin": ["skin"],
                "small_intestine": ["small intestine"],
                "smooth_muscle": ["smooth muscle"],
                "soft_tissue": ["soft tissue"],
                "spleen": ["spleen"],
                "stomach": ["stomach"],
                "testis": ["testis"],
                "thymus": ["thymus"],
                "thyroid_gland": ["thyroid gland"],
                "tongue": ["tongue"],
                "tonsil": ["tonsil"],
                "urinary_bladder": ["urinary bladder"],
                "vagina": ["vagina"]
            },
            "blood": {
                "t_cell": ["t-cell", "t cell"],
                "b_cell": ["b-cell", "b cell"],
                "nk_cell": ["nk-cell", "nk cell", "natural killer"],
                "monocyte": ["monocyte"],
                "neutrophil": ["neutrophil"],
                "eosinophil": ["eosinophil"],
                "basophil": ["basophil"],
                "dendritic_cell": ["dendritic cell"]
            },
            "brain": {
                "cerebellum": ["cerebellum"],
                "cerebral_cortex": ["cerebral cortex", "cortex"],
                "hippocampus": ["hippocampus", "hippocampal formation"],
                "hypothalamus": ["hypothalamus"],
                "amygdala": ["amygdala"],
                "brainstem": ["brainstem", "brain stem"],
                "thalamus": ["thalamus"]
            },
            "single_cell": {
                "t_cell": ["t-cell", "t cell"],
                "b_cell": ["b-cell", "b cell"],
                "hepatocyte": ["hepatocyte"],
                "neuron": ["neuron"],
                "astrocyte": ["astrocyte"],
                "fibroblast": ["fibroblast"]
            }
        }

    def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        gene_name = arguments.get("gene_name")
        source_type = (arguments.get("source_type") or "").lower()
        source_name = (arguments.get("source_name") or "").lower().replace(' ', '_').replace('-', '_')
        
        if not gene_name:
            return {"error": "Parameter 'gene_name' is required"}
        if not source_type:
            return {"error": "Parameter 'source_type' is required"}
        if not source_name:
            return {"error": "Parameter 'source_name' is required"}
        
        # Validate source type
        if source_type not in self.source_column_mappings:
            available_types = ", ".join(self.source_column_mappings.keys())
            return {"error": f"Invalid source_type '{source_type}'. Available types: {available_types}"}
        
        # Enhanced validation with intelligent recommendations
        if source_name not in self.source_name_mappings[source_type]:
            available_sources = list(self.source_name_mappings[source_type].keys())
            
            # Find similar source names (fuzzy matching)
            similar_sources = []
            source_keywords = source_name.replace('_', ' ').split()
            
            for valid_source in available_sources:
                # Direct substring matching
                if (source_name.lower() in valid_source.lower() or 
                    valid_source.lower() in source_name.lower()):
                    similar_sources.append(valid_source)
                    continue
                
                # Check with underscores removed/normalized
                normalized_input = source_name.lower().replace('_', '').replace(' ', '')
                normalized_valid = valid_source.lower().replace('_', '').replace(' ', '')
                if (normalized_input in normalized_valid or 
                    normalized_valid in normalized_input):
                    similar_sources.append(valid_source)
                    continue
                    
                # Check individual keywords
                for keyword in source_keywords:
                    if (keyword.lower() in valid_source.lower() or 
                        valid_source.lower() in keyword.lower()):
                        similar_sources.append(valid_source)
                        break
            
            error_msg = f"Invalid source_name '{source_name}' for source_type '{source_type}'. "
            if similar_sources:
                error_msg += f"Similar options: {similar_sources[:3]}. "
            error_msg += f"All available sources for '{source_type}': {available_sources}"
            return {"error": error_msg}
        
        try:
            # Get the correct API column
            api_column = self.source_column_mappings[source_type]
            columns = f"g,gs,{api_column}"
            
            # Call the search API
            response_data = self._make_api_request(gene_name, columns)
            
            if "error" in response_data:
                return response_data
            
            if not response_data or len(response_data) == 0:
                return {
                    "gene_name": gene_name,
                    "source_type": source_type,
                    "source_name": source_name,
                    "expression_value": "N/A",
                    "status": "Gene not found"
                }
            
            # Get the first result
            gene_data = response_data[0]
            
            # Extract expression data from the API response
            expression_value = "N/A"
            available_sources = []
            
            # Get the expression data dictionary for this source type
            api_field_name = self.api_response_fields[source_type]
            expression_data = gene_data.get(api_field_name)
            
            if expression_data and isinstance(expression_data, dict):
                available_sources = list(expression_data.keys())
                
                # Get possible names for this source
                possible_names = self.source_name_mappings[source_type][source_name]
                
                # Try to find a matching source name in the response
                for source_key in expression_data.keys():
                    source_key_lower = source_key.lower()
                    for possible_name in possible_names:
                        if possible_name.lower() in source_key_lower or source_key_lower in possible_name.lower():
                            expression_value = expression_data[source_key]
                            break
                    if expression_value != "N/A":
                        break
                
                # If exact match not found, look for partial matches
                if expression_value == "N/A":
                    source_keywords = source_name.replace('_', ' ').split()
                    for source_key in expression_data.keys():
                        source_key_lower = source_key.lower()
                        for keyword in source_keywords:
                            if keyword in source_key_lower:
                                expression_value = expression_data[source_key]
                                break
                        if expression_value != "N/A":
                            break
            
            # Categorize expression level
            expression_level = "unknown"
            if expression_value != "N/A":
                try:
                    val = float(expression_value)
                    if val > 50:
                        expression_level = "very high"
                    elif val > 10:
                        expression_level = "high"
                    elif val > 1:
                        expression_level = "medium"
                    elif val > 0.1:
                        expression_level = "low"
                    else:
                        expression_level = "very low"
                except (ValueError, TypeError):
                    expression_level = "unknown"
            
            return {
                "gene_name": gene_data.get("Gene", gene_name),
                "gene_synonym": gene_data.get("Gene synonym", ""),
                "source_type": source_type,
                "source_name": source_name,
                "expression_value": expression_value,
                "expression_level": expression_level,
                "expression_unit": "nTPM",
                "column_queried": api_column,
                "available_sources": available_sources[:10] if len(available_sources) > 10 else available_sources,
                "total_available_sources": len(available_sources),
                "status": "success" if expression_value != "N/A" else "no_expression_data_for_source"
            }
            
        except Exception as e:
            return {
                "error": f"Failed to retrieve RNA expression data: {str(e)}",
                "gene_name": gene_name,
                "source_type": source_type,
                "source_name": source_name
            }


class HPAGetSubcellularLocationTool(HPASearchApiTool):
    """
    Get annotated subcellular locations for a protein using optimized columns parameter.
    Uses scml (main location) and scal (additional location) columns for efficient queries.
    """
    def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        gene_name = arguments.get("gene_name")
        if not gene_name:
            return {"error": "Parameter 'gene_name' is required"}
        
        # Use specific columns for subcellular location data
        result = self._make_api_request(gene_name, "g,gs,scml,scal")
        
        if "error" in result:
            return result
        
        if not result:
            return {"error": "No subcellular location data found"}
        
        gene_data = result[0]
        
        # Parse main and additional locations
        main_location = gene_data.get("Subcellular main location", "")
        additional_location = gene_data.get("Subcellular additional location", "")
        
        # Handle different data types (string or list)
        if isinstance(main_location, list):
            main_locations = main_location
        elif isinstance(main_location, str):
            main_locations = [loc.strip() for loc in main_location.split(';') if loc.strip()] if main_location else []
        else:
            main_locations = []
            
        if isinstance(additional_location, list):
            additional_locations = additional_location
        elif isinstance(additional_location, str):
            additional_locations = [loc.strip() for loc in additional_location.split(';') if loc.strip()] if additional_location else []
        else:
            additional_locations = []
        
        return {
            "gene_name": gene_data.get("Gene", gene_name),
            "gene_synonym": gene_data.get("Gene synonym", ""),
            "main_locations": main_locations,
            "additional_locations": additional_locations,
            "total_locations": len(main_locations) + len(additional_locations),
            "location_summary": self._generate_location_summary(main_locations, additional_locations)
        }
    
    def _generate_location_summary(self, main_locs: List[str], add_locs: List[str]) -> str:
        """Generate a summary of subcellular locations"""
        if not main_locs and not add_locs:
            return "No subcellular location data available"
        
        summary_parts = []
        if main_locs:
            summary_parts.append(f"Primary: {', '.join(main_locs)}")
        if add_locs:
            summary_parts.append(f"Additional: {', '.join(add_locs)}")
        
        return "; ".join(summary_parts)


# --- Existing Tools (Updated with improvements) ---

class HPASearchGenesTool(HPASearchApiTool):
    """
    Search for matching genes by gene name, keywords, or cell line names and return Ensembl ID list.
    This is the entry tool for many query workflows.
    """
    def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        search_query = arguments.get("search_query")
        if not search_query:
            return {"error": "Parameter 'search_query' is required"}
        
        # 'g' for Gene name, 'gs' for Gene synonym, 'eg' for Ensembl ID
        columns = "g,gs,eg"
        result = self._make_api_request(search_query, columns)

        if "error" in result:
            return result
        
        if not result or not isinstance(result, list):
            return {"error": f"No matching genes found for query '{search_query}'"}
            
        formatted_results = []
        for gene in result:
            gene_synonym = gene.get("Gene synonym", "")
            if isinstance(gene_synonym, str):
                synonyms = gene_synonym.split(', ') if gene_synonym else []
            elif isinstance(gene_synonym, list):
                synonyms = gene_synonym
            else:
                synonyms = []
            
            formatted_results.append({
                "gene_name": gene.get("Gene"),
                "ensembl_id": gene.get("Ensembl"),
                "gene_synonyms": synonyms
            })
        
        return {
            "search_query": search_query,
            "match_count": len(formatted_results),
            "genes": formatted_results
        }


class HPAGetComparativeExpressionTool(HPASearchApiTool):
    """
    Compare gene expression levels in specific cell lines and healthy tissues.
    Get expression data for comparison by gene name and cell line name.
    """
    def __init__(self, tool_config):
        super().__init__(tool_config)
        # Mapping of common cell lines to their column identifiers
        self.cell_line_columns = {
            "ishikawa": "cell_RNA_ishikawa_heraklio",
            "hela": "cell_RNA_hela",
            "mcf7": "cell_RNA_mcf7",
            "a549": "cell_RNA_a549",
            "hepg2": "cell_RNA_hepg2",
            "jurkat": "cell_RNA_jurkat",
            "pc3": "cell_RNA_pc3",
            "rh30": "cell_RNA_rh30",
            "siha": "cell_RNA_siha",
            "u251": "cell_RNA_u251"
        }

    def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        gene_name = arguments.get("gene_name")
        cell_line = (arguments.get("cell_line") or "").lower()
        
        if not gene_name:
            return {"error": "Parameter 'gene_name' is required"}
        if not cell_line:
            return {"error": "Parameter 'cell_line' is required"}
        
        # Enhanced validation with intelligent recommendations
        cell_column = self.cell_line_columns.get(cell_line)
        if not cell_column:
            available_lines = list(self.cell_line_columns.keys())
            
            # Find similar cell line names
            similar_lines = []
            for valid_line in available_lines:
                if cell_line in valid_line or valid_line in cell_line:
                    similar_lines.append(valid_line)
            
            error_msg = f"Unsupported cell_line '{cell_line}'. "
            if similar_lines:
                error_msg += f"Similar options: {similar_lines}. "
            error_msg += f"All supported cell lines: {available_lines}"
            return {"error": error_msg}
        
        # Request expression data for the cell line
        cell_columns = f"g,gs,{cell_column}"
        cell_result = self._make_api_request(gene_name, cell_columns)
        if "error" in cell_result:
            return cell_result
        
        # Request expression data for healthy tissues
        tissue_columns = "g,gs,rnatsm"
        tissue_result = self._make_api_request(gene_name, tissue_columns)
        if "error" in tissue_result:
            return tissue_result
        
        # Format the result
        if not cell_result or not tissue_result:
            return {"error": "No expression data found"}
        
        # Extract the first matching gene data
        cell_data = cell_result[0] if isinstance(cell_result, list) and cell_result else {}
        tissue_data = tissue_result[0] if isinstance(tissue_result, list) and tissue_result else {}
        
        return {
            "gene_name": gene_name,
            "gene_symbol": cell_data.get("Gene", gene_name),
            "gene_synonym": cell_data.get("Gene synonym", ""),
            "cell_line": cell_line,
            "cell_line_expression": cell_data.get(cell_column, "N/A"),
            "healthy_tissue_expression": tissue_data.get("RNA tissue specific nTPM", "N/A"),
            "expression_unit": "nTPM (normalized Transcripts Per Million)",
            "comparison_summary": self._generate_comparison_summary(
                cell_data.get(cell_column), 
                tissue_data.get("RNA tissue specific nTPM")
            )
        }
    
    def _generate_comparison_summary(self, cell_expr, tissue_expr) -> str:
        """Generate expression level comparison summary"""
        try:
            cell_val = float(cell_expr) if cell_expr and cell_expr != "N/A" else None
            tissue_val = float(tissue_expr) if tissue_expr and tissue_expr != "N/A" else None
            
            if cell_val is None or tissue_val is None:
                return "Insufficient data for comparison"
            
            if cell_val > tissue_val * 2:
                return f"Expression significantly higher in cell line ({cell_val:.2f} vs {tissue_val:.2f})"
            elif tissue_val > cell_val * 2:
                return f"Expression significantly higher in healthy tissues ({tissue_val:.2f} vs {cell_val:.2f})"
            else:
                return f"Expression levels similar (cell line: {cell_val:.2f}, healthy tissues: {tissue_val:.2f})"
        except:
            return "Failed to calculate expression level comparison"


class HPAGetDiseaseExpressionTool(HPASearchApiTool):
    """
    Get expression data for a gene in specific diseases and tissues.
    Get related expression information by gene name, tissue type, and disease name.
    """
    def __init__(self, tool_config):
        super().__init__(tool_config)
        # Mapping of common cancer types to their column identifiers
        self.cancer_columns = {
            "brain_cancer": "cancer_RNA_brain_cancer",
            "breast_cancer": "cancer_RNA_breast_cancer", 
            "colon_cancer": "cancer_RNA_colon_cancer",
            "lung_cancer": "cancer_RNA_lung_cancer",
            "liver_cancer": "cancer_RNA_liver_cancer",
            "prostate_cancer": "cancer_RNA_prostate_cancer",
            "kidney_cancer": "cancer_RNA_kidney_cancer",
            "pancreatic_cancer": "cancer_RNA_pancreatic_cancer",
            "stomach_cancer": "cancer_RNA_stomach_cancer",
            "ovarian_cancer": "cancer_RNA_ovarian_cancer"
        }

    def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        gene_name = arguments.get("gene_name")
        tissue_type = (arguments.get("tissue_type") or "").lower() 
        disease_name = (arguments.get("disease_name") or "").lower()
        
        if not gene_name:
            return {"error": "Parameter 'gene_name' is required"}
        if not disease_name:
            return {"error": "Parameter 'disease_name' is required"}
        
        # Enhanced validation with intelligent recommendations
        disease_key = f"{tissue_type}_{disease_name}" if tissue_type else disease_name
        cancer_column = None
        
        # Match cancer type
        for key, column in self.cancer_columns.items():
            if disease_key in key or disease_name in key:
                cancer_column = column
                break
        
        if not cancer_column:
            available_diseases = [k.replace("_", " ") for k in self.cancer_columns.keys()]
            
            # Find similar disease names
            similar_diseases = []
            disease_keywords = disease_name.replace('_', ' ').split()
            
            for valid_disease in available_diseases:
                for keyword in disease_keywords:
                    if keyword in valid_disease.lower() or valid_disease.lower() in keyword:
                        similar_diseases.append(valid_disease)
                        break
            
            error_msg = f"Unsupported disease_name '{disease_name}'. "
            if similar_diseases:
                error_msg += f"Similar options: {similar_diseases[:3]}. "
            error_msg += f"All supported diseases: {available_diseases}"
            return {"error": error_msg}
        
        # Build request columns
        columns = f"g,gs,{cancer_column},rnatsm"
        result = self._make_api_request(gene_name, columns)
        
        if "error" in result:
            return result
        
        if not result:
            return {"error": "No expression data found"}
        
        # Extract the first matching gene data
        gene_data = result[0] if isinstance(result, list) and result else {}
        
        return {
            "gene_name": gene_name,
            "gene_symbol": gene_data.get("Gene", gene_name),
            "gene_synonym": gene_data.get("Gene synonym", ""),
            "tissue_type": tissue_type or "Not specified",
            "disease_name": disease_name,
            "disease_expression": gene_data.get(cancer_column, "N/A"),
            "healthy_expression": gene_data.get("RNA tissue specific nTPM", "N/A"),
            "expression_unit": "nTPM (normalized Transcripts Per Million)",
            "disease_vs_healthy": self._compare_disease_healthy(
                gene_data.get(cancer_column),
                gene_data.get("RNA tissue specific nTPM")
            )
        }
    
    def _compare_disease_healthy(self, disease_expr, healthy_expr) -> str:
        """Compare expression difference between disease and healthy state"""
        try:
            disease_val = float(disease_expr) if disease_expr and disease_expr != "N/A" else None
            healthy_val = float(healthy_expr) if healthy_expr and healthy_expr != "N/A" else None
            
            if disease_val is None or healthy_val is None:
                return "Insufficient data for comparison"
            
            fold_change = disease_val / healthy_val if healthy_val > 0 else float('inf')
            
            if fold_change > 2:
                return f"Disease state expression upregulated {fold_change:.2f} fold"
            elif fold_change < 0.5:
                return f"Disease state expression downregulated {1/fold_change:.2f} fold"
            else:
                return f"Expression level relatively stable (fold change: {fold_change:.2f})"
        except:
            return "Failed to calculate expression difference"


class HPAGetBiologicalProcessTool(HPASearchApiTool):
    """
    Get biological process information related to a gene.
    Get specific biological processes a gene is involved in by gene name.
    """
    def __init__(self, tool_config):
        super().__init__(tool_config)
        # Predefined biological process list
        self.target_processes = [
            'Apoptosis', 'Biological rhythms', 'Cell cycle', 
            'Host-virus interaction', 'Necrosis', 'Transcription', 
            'Transcription regulation'
        ]

    def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        gene_name = arguments.get("gene_name")
        filter_processes = arguments.get("filter_processes", True)
        
        if not gene_name:
            return {"error": "Parameter 'gene_name' is required"}
        
        # Request biological process data for the gene
        columns = "g,gs,upbp"
        result = self._make_api_request(gene_name, columns)
        
        if "error" in result:
            return result
        
        if not result:
            return {"error": "No gene data found"}
        
        # Extract the first matching gene data
        gene_data = result[0] if isinstance(result, list) and result else {}
        
        # Parse biological processes
        biological_processes = gene_data.get("Biological process", "")
        if not biological_processes or biological_processes == "N/A":
            return {
                "gene_name": gene_name,
                "gene_symbol": gene_data.get("Gene", gene_name),
                "gene_synonym": gene_data.get("Gene synonym", ""),
                "biological_processes": [],
                "target_processes_found": [],
                "target_process_names": [],
                "total_processes": 0,
                "target_processes_count": 0
            }
        
        # Split and clean process list - handle both string and list formats
        processes_list = []
        if isinstance(biological_processes, list):
            processes_list = biological_processes
        elif isinstance(biological_processes, str):
            # Usually separated by semicolon or comma
            processes_list = [p.strip() for p in biological_processes.replace(';', ',').split(',') if p.strip()]
        
        # Filter target processes
        target_found = []
        if filter_processes:
            for process in processes_list:
                for target in self.target_processes:
                    if target.lower() in process.lower():
                        target_found.append({
                            "target_process": target,
                            "full_description": process
                        })
        
        return {
            "gene_name": gene_name,
            "gene_symbol": gene_data.get("Gene", gene_name),
            "gene_synonym": gene_data.get("Gene synonym", ""),
            "biological_processes": processes_list,
            "target_processes_found": target_found,
            "target_process_names": [tp["target_process"] for tp in target_found],
            "total_processes": len(processes_list),
            "target_processes_count": len(target_found)
        }


class HPAGetCancerPrognosticsTool(HPAJsonApiTool):
    """
    Get prognostic value of a gene across various cancers.
    Uses the efficient JSON API to retrieve cancer prognostic data.
    """
    def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        ensembl_id = arguments.get("ensembl_id")
        if not ensembl_id:
            return {"error": "Parameter 'ensembl_id' is required"}
        
        data = self._make_api_request(ensembl_id)
        if "error" in data:
            return data
            
        prognostics = []
        for key, value in data.items():
            if key.startswith("Cancer prognostics") and isinstance(value, dict):
                cancer_type = key.replace("Cancer prognostics - ", "").strip()
                if value and value.get('is_prognostic'):
                    prognostics.append({
                        "cancer_type": cancer_type,
                        "prognostic_type": value.get("prognostic type", "Unknown"),
                        "p_value": value.get("p_val", "N/A"),
                        "is_prognostic": value.get("is_prognostic", False)
                    })
        
        return {
            "ensembl_id": ensembl_id,
            "gene": data.get("Gene", "Unknown"),
            "gene_synonym": data.get("Gene synonym", ""),
            "prognostic_cancers_count": len(prognostics),
            "prognostic_summary": prognostics if prognostics else "No significant prognostic value found in the analyzed cancers.",
            "note": "Prognostic value indicates whether high/low expression of this gene correlates with patient survival in specific cancer types."
        }


class HPAGetProteinInteractionsTool(HPASearchApiTool):
    """
    Get protein-protein interaction partners for a gene.
    Uses search API to retrieve interaction data.
    """
    def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        gene_name = arguments.get("gene_name")
        if not gene_name:
            return {"error": "Parameter 'gene_name' is required"}
        
        # Use 'ppi' column to retrieve protein-protein interactions
        columns = "g,gs,ppi"
        result = self._make_api_request(gene_name, columns)

        if "error" in result:
            return result
        
        if not result or not isinstance(result, list):
            return {"error": f"No interaction data found for gene '{gene_name}'"}

        gene_data = result[0]
        interactions_str = gene_data.get("Protein-protein interaction", "")
        
        if not interactions_str or interactions_str == "N/A":
            return {
                "gene": gene_data.get("Gene", gene_name),
                "gene_synonym": gene_data.get("Gene synonym", ""),
                "interactions": "No interaction data found.",
                "interactor_count": 0,
                "interactors": []
            }

        # Parse interaction string (usually semicolon or comma separated)
        interactors = [i.strip() for i in interactions_str.replace(';', ',').split(',') if i.strip()]
        
        return {
            "gene": gene_data.get("Gene", gene_name),
            "gene_synonym": gene_data.get("Gene synonym", ""),
            "interactor_count": len(interactors),
            "interactors": interactors,
            "interaction_summary": f"Found {len(interactors)} protein interaction partners"
        }


class HPAGetRnaExpressionByTissueTool(HPAJsonApiTool):
    """
    Query RNA expression levels for a gene in specific tissues.
    More precise than general tissue expression queries.
    """
    def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        ensembl_id = arguments.get("ensembl_id")
        tissue_names = arguments.get("tissue_names", [])
        
        if not ensembl_id:
            return {"error": "Parameter 'ensembl_id' is required"}
        if not tissue_names or not isinstance(tissue_names, list):
            # Provide helpful tissue name examples
            example_tissues = ["brain", "liver", "heart", "kidney", "lung", "pancreas", "skin", "muscle"]
            return {"error": f"Parameter 'tissue_names' is required and must be a list. Example: {example_tissues}"}
            
        data = self._make_api_request(ensembl_id)
        if "error" in data:
            return data

        # Get RNA tissue expression data
        rna_data = data.get("RNA tissue specific nTPM", {})
        if not isinstance(rna_data, dict):
            return {"error": "No RNA tissue expression data available for this gene"}
        
        expression_results = {}
        available_tissues = list(rna_data.keys())
        
        for tissue in tissue_names:
            # Case-insensitive matching
            found_tissue = None
            for available_tissue in available_tissues:
                if tissue.lower() in available_tissue.lower() or available_tissue.lower() in tissue.lower():
                    found_tissue = available_tissue
                    break
            
            if found_tissue:
                expression_results[tissue] = {
                    "matched_tissue": found_tissue,
                    "expression_value": rna_data[found_tissue],
                    "expression_level": self._categorize_expression(rna_data[found_tissue])
                }
            else:
                expression_results[tissue] = {
                    "matched_tissue": "Not found",
                    "expression_value": "N/A",
                    "expression_level": "No data"
                }
        
        return {
            "ensembl_id": ensembl_id,
            "gene": data.get("Gene", "Unknown"),
            "gene_synonym": data.get("Gene synonym", ""),
            "expression_unit": "nTPM (normalized Transcripts Per Million)",
            "queried_tissues": tissue_names,
            "tissue_expression": expression_results,
            "available_tissues_sample": available_tissues[:10] if len(available_tissues) > 10 else available_tissues,
            "total_available_tissues": len(available_tissues)
        }
    
    def _categorize_expression(self, expr_value) -> str:
        """Categorize expression level"""
        try:
            val = float(expr_value)
            if val > 50:
                return "Very high"
            elif val > 10:
                return "High"
            elif val > 1:
                return "Medium"
            elif val > 0.1:
                return "Low"
            else:
                return "Very low"
        except (ValueError, TypeError):
            return "Unknown"


class HPAGetContextualBiologicalProcessTool:
    """
    Analyze a gene's biological processes in the context of specific tissue or cell line.
    Enhanced with intelligent context validation and recommendation.
    """
    def __init__(self, tool_config):
        super().__init__(tool_config)
        # Define all valid context options
        self.valid_contexts = {
            "tissues": [
                "adipose_tissue", "adrenal_gland", "appendix", "bone_marrow", "brain", "breast", 
                "bronchus", "cerebellum", "cerebral_cortex", "cervix", "colon", "duodenum", 
                "endometrium", "esophagus", "gallbladder", "heart_muscle", "kidney", "liver", 
                "lung", "lymph_node", "ovary", "pancreas", "placenta", "prostate", "rectum", 
                "salivary_gland", "skeletal_muscle", "skin", "small_intestine", "spleen", 
                "stomach", "testis", "thymus", "thyroid_gland", "urinary_bladder", "vagina"
            ],
            "cell_lines": ["hela", "mcf7", "a549", "hepg2", "jurkat", "pc3", "rh30", "siha", "u251"],
            "blood_cells": ["t_cell", "b_cell", "nk_cell", "monocyte", "neutrophil", "eosinophil"],
            "brain_regions": ["cerebellum", "cerebral_cortex", "hippocampus", "hypothalamus", "amygdala"]
        }
        
    def _validate_context(self, context_name: str) -> Dict[str, Any]:
        """Validate context_name and provide intelligent recommendations"""
        context_lower = context_name.lower().replace(' ', '_').replace('-', '_')
        
        # Check all valid contexts
        all_valid = []
        for category, contexts in self.valid_contexts.items():
            all_valid.extend(contexts)
            if context_lower in contexts:
                return {"valid": True, "category": category}
        
        # Find similar contexts (fuzzy matching)
        similar_contexts = []
        context_keywords = context_lower.split('_')
        
        for valid_context in all_valid:
            for keyword in context_keywords:
                if keyword in valid_context.lower() or valid_context.lower() in keyword:
                    similar_contexts.append(valid_context)
                    break
        
        return {
            "valid": False,
            "input": context_name,
            "similar_suggestions": similar_contexts[:5],  # Top 5 suggestions
            "all_tissues": self.valid_contexts["tissues"][:10],  # First 10 tissues
            "all_cell_lines": self.valid_contexts["cell_lines"],
            "total_available": len(all_valid)
        }
        
    def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        gene_name = arguments.get("gene_name")
        context_name = arguments.get("context_name")
        
        if not gene_name:
            return {"error": "Parameter 'gene_name' is required"}
        if not context_name:
            return {"error": "Parameter 'context_name' is required"}
        
        # Validate context_name and provide recommendations if invalid
        validation = self._validate_context(context_name)
        if not validation["valid"]:
            error_msg = f"Invalid context_name '{validation['input']}'. "
            if validation["similar_suggestions"]:
                error_msg += f"Similar options: {validation['similar_suggestions']}. "
            error_msg += f"Available tissues: {validation['all_tissues']}... "
            error_msg += f"Available cell lines: {validation['all_cell_lines']}. "
            error_msg += f"Total {validation['total_available']} contexts available."
            return {"error": error_msg}
        
        try:
            # Step 1: Get gene basic info and Ensembl ID
            search_api = HPASearchApiTool({})
            search_result = search_api._make_api_request(gene_name, "g,gs,eg,upbp")
            
            if "error" in search_result or not search_result:
                return {"error": f"Could not find gene information for '{gene_name}'"}
            
            gene_data = search_result[0] if isinstance(search_result, list) else search_result
            ensembl_id = gene_data.get("Ensembl", "")
            
            if not ensembl_id:
                return {"error": f"Could not find Ensembl ID for gene '{gene_name}'"}
            
            # Step 2: Get biological processes
            biological_processes = gene_data.get("Biological process", "")
            processes_list = []
            if biological_processes and biological_processes != "N/A":
                if isinstance(biological_processes, list):
                    processes_list = biological_processes
                elif isinstance(biological_processes, str):
                    processes_list = [p.strip() for p in biological_processes.replace(';', ',').split(',') if p.strip()]
            
            # Step 3: Get expression in context with improved error handling
            json_api = HPAJsonApiTool({})
            json_data = json_api._make_api_request(ensembl_id)
            
            expression_value = "N/A"
            expression_level = "not expressed"
            context_type = validation["category"].replace('_', ' ').rstrip('s')  # "tissues" -> "tissue"
            
            if "error" not in json_data and json_data:
                # FIXED: Check if rna_data is not None before calling .keys()
                rna_data = json_data.get("RNA tissue specific nTPM")
                if rna_data and isinstance(rna_data, dict):
                    # Try to find matching tissue
                    for tissue_key in rna_data.keys():
                        if context_name.lower() in tissue_key.lower() or tissue_key.lower() in context_name.lower():
                            expression_value = rna_data[tissue_key]
                            break
                
                # If not found in tissues and it's a cell line, try cell line data
                if expression_value == "N/A" and validation["category"] == "cell_lines":
                    context_type = "cell line"
                    cell_line_columns = {
                        "hela": "cell_RNA_hela", "mcf7": "cell_RNA_mcf7", 
                        "a549": "cell_RNA_a549", "hepg2": "cell_RNA_hepg2"
                    }
                    
                    cell_column = cell_line_columns.get(context_name.lower())
                    if cell_column:
                        cell_result = search_api._make_api_request(gene_name, f"g,{cell_column}")
                        if "error" not in cell_result and cell_result:
                            expression_value = cell_result[0].get(cell_column, "N/A")
            
            # Categorize expression level
            try:
                expr_val = float(expression_value) if expression_value != "N/A" else 0
                if expr_val > 10:
                    expression_level = "highly expressed"
                elif expr_val > 1:
                    expression_level = "moderately expressed"
                elif expr_val > 0.1:
                    expression_level = "expressed at low level"
                else:
                    expression_level = "not expressed or very low"
            except (ValueError, TypeError):
                expression_level = "expression level unclear"
            
            # Generate contextual conclusion
            relevance = "may be functionally relevant" if "expressed" in expression_level and "not" not in expression_level else "is likely not functionally relevant"
            
            conclusion = f"Gene {gene_name} is involved in {len(processes_list)} biological processes. It is {expression_level} in {context_name} ({expression_value} nTPM), suggesting its functional roles {relevance} in this {context_type} context."
            
            return {
                "gene": gene_data.get("Gene", gene_name),
                "gene_synonym": gene_data.get("Gene synonym", ""),
                "ensembl_id": ensembl_id,
                "context": context_name,
                "context_type": context_type,
                "context_category": validation["category"],
                "expression_in_context": f"{expression_value} nTPM",
                "expression_level": expression_level,
                "total_biological_processes": len(processes_list),
                "biological_processes": processes_list[:10] if len(processes_list) > 10 else processes_list,
                "contextual_conclusion": conclusion,
                "functional_relevance": relevance
            }
            
        except Exception as e:
            return {"error": f"Failed to perform contextual analysis: {str(e)}"}


# --- Keep existing comprehensive gene details tool for images ---

class HPAGetGenePageDetailsTool(HPAXmlApiTool):
    """
    Get detailed information about a gene page, including images, protein expression, antibody data, etc.
    Get the most comprehensive data by parsing HPA's single gene XML endpoint.
    Enhanced version with improved image extraction and comprehensive data parsing based on optimization plan.
    """
    def __init__(self, tool_config):
        super().__init__(tool_config)

    def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        ensembl_id = arguments.get("ensembl_id")
        include_images = arguments.get("include_images", True)
        include_antibodies = arguments.get("include_antibodies", True)
        include_expression = arguments.get("include_expression", True)
        
        if not ensembl_id:
            return {"error": "Parameter 'ensembl_id' is required"}
        
        try:
            root = self._make_api_request(ensembl_id)
            return self._parse_gene_xml(root, ensembl_id, include_images, include_antibodies, include_expression)
            
        except Exception as e:
            return {"error": str(e)}

    def _parse_gene_xml(self, root: ET.Element, ensembl_id: str, include_images: bool, 
                       include_antibodies: bool, include_expression: bool) -> Dict[str, Any]:
        """Parse gene XML data comprehensively based on actual HPA XML schema"""
        result = {
            "ensembl_id": ensembl_id,
            "gene_name": "",
            "gene_description": "",
            "chromosome_location": "",
            "uniprot_ids": [],
            "summary": {}
        }
        
        # Extract basic gene information from entry element
        entry_elem = root.find('.//entry')
        if entry_elem is not None:
            # Gene name
            name_elem = entry_elem.find('name')
            if name_elem is not None:
                result["gene_name"] = name_elem.text or ""
            
            # Gene synonyms
            synonyms = []
            for synonym_elem in entry_elem.findall('synonym'):
                if synonym_elem.text:
                    synonyms.append(synonym_elem.text)
            result["gene_synonyms"] = synonyms
            
            # Extract Uniprot IDs from identifier/xref elements
            identifier_elem = entry_elem.find('identifier')
            if identifier_elem is not None:
                for xref in identifier_elem.findall('xref'):
                    if xref.get('db') == 'Uniprot/SWISSPROT':
                        result["uniprot_ids"].append(xref.get('id', ''))
            
            # Extract protein classes
            protein_classes = []
            protein_classes_elem = entry_elem.find('proteinClasses')
            if protein_classes_elem is not None:
                for pc in protein_classes_elem.findall('proteinClass'):
                    class_name = pc.get('name', '')
                    if class_name:
                        protein_classes.append(class_name)
            result["protein_classes"] = protein_classes
        
        # Extract image information with enhanced parsing
        if include_images:
            result["ihc_images"] = self._extract_ihc_images(root)
            result["if_images"] = self._extract_if_images(root)
        
        # Extract antibody information
        if include_antibodies:
            result["antibodies"] = self._extract_antibodies(root)
        
        # Extract expression information
        if include_expression:
            result["expression_summary"] = self._extract_expression_summary(root)
            result["tissue_expression"] = self._extract_tissue_expression(root)
            result["cell_line_expression"] = self._extract_cell_line_expression(root)
        
        # Extract summary statistics
        result["summary"] = {
            "total_antibodies": len(result.get("antibodies", [])),
            "total_ihc_images": len(result.get("ihc_images", [])),
            "total_if_images": len(result.get("if_images", [])),
            "tissues_with_expression": len(result.get("tissue_expression", [])),
            "cell_lines_with_expression": len(result.get("cell_line_expression", []))
        }
        
        return result

    def _extract_ihc_images(self, root: ET.Element) -> List[Dict[str, Any]]:
        """Extract tissue immunohistochemistry (IHC) images based on actual HPA XML structure"""
        images = []
        
        # Find tissueExpression elements which contain IHC images
        for tissue_expr in root.findall('.//tissueExpression'):
            # Extract selected images from tissueExpression
            for image_elem in tissue_expr.findall('.//image'):
                image_type = image_elem.get('imageType', '')
                if image_type == 'selected':
                    tissue_elem = image_elem.find('tissue')
                    image_url_elem = image_elem.find('imageUrl')
                    
                    if tissue_elem is not None and image_url_elem is not None:
                        tissue_name = tissue_elem.text or ''
                        organ = tissue_elem.get('organ', '')
                        ontology_terms = tissue_elem.get('ontologyTerms', '')
                        image_url = image_url_elem.text or ''
                        
                        images.append({
                            "image_type": "Immunohistochemistry",
                            "tissue_name": tissue_name,
                            "organ": organ,
                            "ontology_terms": ontology_terms,
                            "image_url": image_url,
                            "selected": True
                        })
        
        return images

    def _extract_if_images(self, root: ET.Element) -> List[Dict[str, Any]]:
        """Extract subcellular immunofluorescence (IF) images based on actual HPA XML structure"""
        images = []
        
        # Look for subcellular expression data (IF images are typically in subcellular sections)
        for subcell_expr in root.findall('.//subcellularExpression'):
            # Extract subcellular location images
            for image_elem in subcell_expr.findall('.//image'):
                image_type = image_elem.get('imageType', '')
                if image_type == 'selected':
                    location_elem = image_elem.find('location')
                    image_url_elem = image_elem.find('imageUrl')
                    
                    if location_elem is not None and image_url_elem is not None:
                        location_name = location_elem.text or ''
                        image_url = image_url_elem.text or ''
                        
                        images.append({
                            "image_type": "Immunofluorescence",
                            "subcellular_location": location_name,
                            "image_url": image_url,
                            "selected": True
                })
        
        return images

    def _extract_antibodies(self, root: ET.Element) -> List[Dict[str, Any]]:
        """Extract antibody information from actual HPA XML structure"""
        antibodies_data = []
        
        # Look for antibody references in various expression sections
        antibody_ids = set()
        
        # Look for antibody references in tissue expression
        for tissue_expr in root.findall('.//tissueExpression'):
            for elem in tissue_expr.iter():
                if 'antibody' in elem.tag.lower() or elem.get('antibody'):
                    antibody_id = elem.get('antibody') or elem.text
                    if antibody_id:
                        antibody_ids.add(antibody_id)
        
        # Create basic antibody info for found IDs
        for antibody_id in antibody_ids:
            antibodies_data.append({
                "antibody_id": antibody_id,
                "source": "HPA",
                "applications": ["IHC", "IF"],
                "validation_status": "Available"
            })
        
        # If no specific antibody IDs found, create a placeholder
        if not antibodies_data:
            antibodies_data.append({
                "antibody_id": "HPA_antibody",
                "source": "HPA",
                "applications": ["IHC", "IF"],
                "validation_status": "Available"
            })
        
        return antibodies_data

    def _extract_expression_summary(self, root: ET.Element) -> Dict[str, Any]:
        """Extract expression summary information from actual HPA XML structure"""
        summary = {
            "tissue_specificity": "",
            "subcellular_location": [],
            "protein_class": [],
            "predicted_location": "",
            "tissue_expression_summary": "",
            "subcellular_expression_summary": ""
        }
        
        # Extract predicted location
        predicted_location_elem = root.find('.//predictedLocation')
        if predicted_location_elem is not None:
            summary["predicted_location"] = predicted_location_elem.text or ""
        
        # Extract tissue expression summary
        tissue_expr_elem = root.find('.//tissueExpression')
        if tissue_expr_elem is not None:
            tissue_summary_elem = tissue_expr_elem.find('summary')
            if tissue_summary_elem is not None:
                summary["tissue_expression_summary"] = tissue_summary_elem.text or ""
        
        # Extract subcellular expression summary
        subcell_expr_elem = root.find('.//subcellularExpression')
        if subcell_expr_elem is not None:
            subcell_summary_elem = subcell_expr_elem.find('summary')
            if subcell_summary_elem is not None:
                summary["subcellular_expression_summary"] = subcell_summary_elem.text or ""
        
        return summary

    def _extract_tissue_expression(self, root: ET.Element) -> List[Dict[str, Any]]:
        """Extract detailed tissue expression data from actual HPA XML structure"""
        tissue_data = []
        
        # Extract from tissueExpression data elements
        for tissue_expr in root.findall('.//tissueExpression'):
            for data_elem in tissue_expr.findall('.//data'):
                tissue_elem = data_elem.find('tissue')
                level_elem = data_elem.find('level')
                
                if tissue_elem is not None:
                    tissue_info = {
                        "tissue_name": tissue_elem.text or '',
                        "organ": tissue_elem.get('organ', ''),
                        "expression_level": "",
                    }
                    
                    if level_elem is not None:
                        tissue_info["expression_level"] = level_elem.get('type', '') + ': ' + (level_elem.text or '')
                    
                    tissue_data.append(tissue_info)
        
        return tissue_data

    def _extract_cell_line_expression(self, root: ET.Element) -> List[Dict[str, Any]]:
        """Extract cell line expression data from actual HPA XML structure"""
        cell_line_data = []
        
        # Look for cell line expression in subcellular expression
        for subcell_expr in root.findall('.//subcellularExpression'):
            for data_elem in subcell_expr.findall('.//data'):
                cell_line_elem = data_elem.find('cellLine')
                if cell_line_elem is not None:
                    cell_info = {
                        "cell_line_name": cell_line_elem.get('name', '') or (cell_line_elem.text or ''),
                        "expression_data": []
                    }
                    
                    if cell_info["expression_data"]:
                        cell_line_data.append(cell_info)
        
        return cell_line_data


# --- Legacy/Compatibility Tools ---

class HPAGetGeneJSONTool(HPAJsonApiTool):
    """
    Enhanced legacy tool - Get basic gene information using Ensembl Gene ID.
    Now uses the efficient JSON API instead of search API.
    """
    def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        ensembl_id = arguments.get("ensembl_id")
        if not ensembl_id:
            return {"error": "Parameter 'ensembl_id' is required"}
        
        # Use JSON API to get comprehensive information
        data = self._make_api_request(ensembl_id)
        
        if "error" in data:
            return data
        
        # Convert to response similar to original JSON format for compatibility
        return {
            "Ensembl": ensembl_id,
            "Gene": data.get("Gene", ""),
            "Gene synonym": data.get("Gene synonym", ""),
            "Uniprot": data.get("Uniprot", ""),
            "Biological process": data.get("Biological process", ""),
            "RNA tissue specific nTPM": data.get("RNA tissue specific nTPM", "")
        }


class HPAGetGeneXMLTool(HPASearchApiTool):
    """
    Legacy tool - Get gene TSV format data (alternative to XML).
    """
    def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        ensembl_id = arguments.get("ensembl_id")
        if not ensembl_id:
            return {"error": "Parameter 'ensembl_id' is required"}
        
        # Use TSV format to get detailed data
        columns = "g,gs,up,upbp,rnatsm,cell_RNA_a549,cell_RNA_hela"
        result = self._make_api_request(ensembl_id, columns, format_type="tsv")
        
        if "error" in result:
            return result
        
        return {"tsv_data": result.get("tsv_data", "")}


class HPAGetComprehensiveBiologicalProcessTool(HPASearchApiTool):
    """
    Comprehensive biological process analysis tool that leverages HPAGetBiologicalProcessTool.
    Provides enhanced functionality including process categorization, pathway analysis, 
    comparative analysis, and functional insights for genes.
    """
    def __init__(self, tool_config):
        super().__init__(tool_config)
        
        # Enhanced biological process categories for better organization
        self.process_categories = {
            "cell_cycle": {
                "keywords": ["cell cycle", "mitosis", "meiosis", "cell division", "proliferation"],
                "description": "Processes related to cell division and growth control",
                "priority": "high"
            },
            "apoptosis": {
                "keywords": ["apoptosis", "programmed cell death", "cell death"],
                "description": "Programmed cell death processes",
                "priority": "high"
            },
            "transcription": {
                "keywords": ["transcription", "transcription regulation", "gene expression"],
                "description": "Gene expression and transcriptional control",
                "priority": "high"
            },
            "metabolism": {
                "keywords": ["metabolism", "metabolic", "biosynthesis", "catabolism"],
                "description": "Metabolic and biosynthetic processes",
                "priority": "medium"
            },
            "signaling": {
                "keywords": ["signaling", "signal transduction", "receptor", "pathway"],
                "description": "Cell signaling and communication",
                "priority": "high"
            },
            "immune": {
                "keywords": ["immune", "immunity", "inflammation", "defense"],
                "description": "Immune system and defense mechanisms",
                "priority": "medium"
            },
            "development": {
                "keywords": ["development", "differentiation", "morphogenesis", "growth"],
                "description": "Developmental and differentiation processes",
                "priority": "medium"
            },
            "stress_response": {
                "keywords": ["stress", "response", "oxidative", "heat shock"],
                "description": "Cellular stress response mechanisms",
                "priority": "medium"
            },
            "transport": {
                "keywords": ["transport", "secretion", "import", "export"],
                "description": "Cellular transport and secretion",
                "priority": "low"
            },
            "dna_repair": {
                "keywords": ["dna repair", "dna damage", "recombination"],
                "description": "DNA repair and maintenance",
                "priority": "high"
            }
        }
        
        # Critical biological processes for disease relevance
        self.critical_processes = [
            "apoptosis", "cell cycle", "dna repair", "transcription regulation",
            "signal transduction", "immune response", "metabolism"
        ]

    def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive biological process analysis with enhanced categorization and insights.
        
        Args:
            gene_name (str): Name of the gene to analyze
            include_categorization (bool): Whether to categorize processes by function
            include_pathway_analysis (bool): Whether to analyze pathway involvement
            include_comparative_analysis (bool): Whether to compare with other genes
            max_processes (int): Maximum number of processes to return (default: 50)
            filter_critical_only (bool): Whether to focus only on critical processes
        
        Returns:
            Dict containing comprehensive biological process analysis
        """
        gene_name = arguments.get("gene_name")
        include_categorization = arguments.get("include_categorization", True)
        include_pathway_analysis = arguments.get("include_pathway_analysis", True)
        include_comparative_analysis = arguments.get("include_comparative_analysis", False)
        max_processes = arguments.get("max_processes", 50)
        filter_critical_only = arguments.get("filter_critical_only", False)
        
        if not gene_name:
            return {"error": "Parameter 'gene_name' is required"}
        
        try:
            # Step 1: Get basic biological process data using the existing tool
            basic_tool = HPAGetBiologicalProcessTool({})
            basic_result = basic_tool.run({"gene_name": gene_name, "filter_processes": False})
            
            if "error" in basic_result:
                return basic_result
            
            # Step 2: Enhanced process analysis
            enhanced_analysis = self._enhance_process_analysis(
                basic_result, 
                include_categorization, 
                include_pathway_analysis,
                max_processes,
                filter_critical_only
            )
            
            # Step 3: Comparative analysis if requested
            comparative_data = {}
            if include_comparative_analysis:
                comparative_data = self._perform_comparative_analysis(gene_name, basic_result)
            
            # Step 4: Generate functional insights
            functional_insights = self._generate_functional_insights(enhanced_analysis, basic_result)
            
            # Step 5: Compile comprehensive result
            result = {
                "gene_name": basic_result.get("gene_name", gene_name),
                "gene_symbol": basic_result.get("gene_symbol", gene_name),
                "gene_synonym": basic_result.get("gene_synonym", ""),
                "analysis_summary": {
                    "total_processes": basic_result.get("total_processes", 0),
                    "categorized_processes": len(enhanced_analysis.get("categorized_processes", {})),
                    "critical_processes_found": len(enhanced_analysis.get("critical_processes", [])),
                    "pathway_involvement": len(enhanced_analysis.get("pathway_analysis", {}))
                },
                "biological_processes": basic_result.get("biological_processes", [])[:max_processes],
                "enhanced_analysis": enhanced_analysis,
                "functional_insights": functional_insights,
                "comparative_analysis": comparative_data,
                "metadata": {
                    "analysis_timestamp": self._get_timestamp(),
                    "analysis_version": "2.0",
                    "data_source": "Human Protein Atlas",
                    "confidence_level": self._calculate_confidence_level(basic_result)
                }
            }
            
            return result
            
        except Exception as e:
            return {
                "error": f"Failed to perform comprehensive biological process analysis: {str(e)}",
                "gene_name": gene_name
            }

    def _enhance_process_analysis(self, basic_result: Dict[str, Any], 
                                include_categorization: bool, 
                                include_pathway_analysis: bool,
                                max_processes: int,
                                filter_critical_only: bool) -> Dict[str, Any]:
        """Enhance basic process analysis with categorization and pathway information"""
        processes = basic_result.get("biological_processes", [])
        
        # Filter critical processes if requested
        if filter_critical_only:
            processes = [p for p in processes if any(cp.lower() in p.lower() for cp in self.critical_processes)]
        
        # Limit processes
        processes = processes[:max_processes]
        
        enhanced_analysis = {
            "categorized_processes": {},
            "critical_processes": [],
            "pathway_analysis": {},
            "process_complexity_score": 0
        }
        
        if include_categorization:
            enhanced_analysis["categorized_processes"] = self._categorize_processes(processes)
            enhanced_analysis["critical_processes"] = self._identify_critical_processes(processes)
        
        if include_pathway_analysis:
            enhanced_analysis["pathway_analysis"] = self._analyze_pathway_involvement(processes)
        
        # Calculate process complexity score
        enhanced_analysis["process_complexity_score"] = self._calculate_complexity_score(processes)
        
        return enhanced_analysis

    def _categorize_processes(self, processes: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize biological processes by functional groups"""
        categorized = {}
        
        for process in processes:
            process_lower = process.lower()
            categorized_in = []
            
            for category, config in self.process_categories.items():
                for keyword in config["keywords"]:
                    if keyword.lower() in process_lower:
                        categorized_in.append({
                            "category": category,
                            "description": config["description"],
                            "priority": config["priority"],
                            "confidence": self._calculate_category_confidence(process, keyword)
                        })
                        break
            
            if categorized_in:
                # Sort by priority and confidence
                categorized_in.sort(key=lambda x: (x["priority"] == "high", x["confidence"]), reverse=True)
                best_category = categorized_in[0]["category"]
                
                if best_category not in categorized:
                    categorized[best_category] = []
                
                categorized[best_category].append({
                    "process": process,
                    "category_info": categorized_in[0],
                    "alternative_categories": categorized_in[1:] if len(categorized_in) > 1 else []
                })
            else:
                # Uncategorized processes
                if "uncategorized" not in categorized:
                    categorized["uncategorized"] = []
                
                categorized["uncategorized"].append({
                    "process": process,
                    "category_info": {
                        "category": "uncategorized",
                        "description": "Process not fitting into standard categories",
                        "priority": "low",
                        "confidence": 0.0
                    }
                })
        
        return categorized

    def _identify_critical_processes(self, processes: List[str]) -> List[Dict[str, Any]]:
        """Identify critical biological processes that are essential for cell function"""
        critical_found = []
        
        for process in processes:
            process_lower = process.lower()
            for critical in self.critical_processes:
                if critical.lower() in process_lower:
                    critical_found.append({
                        "process": process,
                        "critical_type": critical,
                        "importance": "essential",
                        "disease_relevance": self._assess_disease_relevance(critical)
                    })
                    break
        
        return critical_found

    def _analyze_pathway_involvement(self, processes: List[str]) -> Dict[str, Any]:
        """Analyze pathway involvement based on biological processes"""
        pathway_keywords = {
            "cell_cycle_pathway": ["cell cycle", "mitosis", "meiosis"],
            "apoptosis_pathway": ["apoptosis", "programmed cell death"],
            "dna_repair_pathway": ["dna repair", "dna damage"],
            "metabolic_pathway": ["metabolism", "biosynthesis", "catabolism"],
            "signaling_pathway": ["signaling", "signal transduction"],
            "immune_pathway": ["immune", "inflammation"],
            "transcription_pathway": ["transcription", "gene expression"]
        }
        
        pathway_involvement = {}
        
        for pathway, keywords in pathway_keywords.items():
            involvement_score = 0
            matching_processes = []
            
            for process in processes:
                process_lower = process.lower()
                for keyword in keywords:
                    if keyword.lower() in process_lower:
                        involvement_score += 1
                        matching_processes.append(process)
                        break
            
            if involvement_score > 0:
                pathway_involvement[pathway] = {
                    "involvement_score": involvement_score,
                    "matching_processes": matching_processes,
                    "pathway_confidence": min(involvement_score / len(keywords), 1.0)
                }
        
        return pathway_involvement

    def _perform_comparative_analysis(self, gene_name: str, basic_result: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comparative analysis with similar genes"""
        try:
            # Get protein interactions to find related genes
            interaction_tool = HPAGetProteinInteractionsTool({})
            interaction_result = interaction_tool.run({"gene_name": gene_name})
            
            comparative_data = {
                "related_genes": [],
                "functional_similarity": {},
                "pathway_overlap": {}
            }
            
            if "error" not in interaction_result and interaction_result.get("interactors"):
                interactors = interaction_result.get("interactors", [])[:5]  # Limit to top 5
                
                for interactor in interactors:
                    try:
                        # Get biological processes for interacting protein
                        interactor_tool = HPAGetBiologicalProcessTool({})
                        interactor_result = interactor_tool.run({"gene_name": interactor, "filter_processes": False})
                        
                        if "error" not in interactor_result:
                            similarity_score = self._calculate_functional_similarity(
                                basic_result.get("biological_processes", []),
                                interactor_result.get("biological_processes", [])
                            )
                            
                            comparative_data["related_genes"].append({
                                "gene_name": interactor,
                                "interaction_type": "protein-protein interaction",
                                "functional_similarity": similarity_score,
                                "shared_processes": self._find_shared_processes(
                                    basic_result.get("biological_processes", []),
                                    interactor_result.get("biological_processes", [])
                                )
                            })
                    except:
                        continue  # Skip if analysis fails for this interactor
                
                # Sort by functional similarity
                comparative_data["related_genes"].sort(key=lambda x: x["functional_similarity"], reverse=True)
            
            return comparative_data
            
        except Exception as e:
            return {"error": f"Comparative analysis failed: {str(e)}"}

    def _generate_functional_insights(self, enhanced_analysis: Dict[str, Any], 
                                   basic_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate functional insights based on biological process analysis"""
        insights = {
            "critical_processes": [],
            "disease_relevant_processes": [],
            "therapeutic_potential": "",
            "research_priorities": [],
            "confidence_assessment": ""
        }
        
        # Extract critical processes with details
        critical_processes = enhanced_analysis.get("critical_processes", [])
        insights["critical_processes"] = critical_processes
        
        # Identify disease-relevant biological processes
        disease_relevant_processes = self._identify_disease_relevant_processes(
            basic_result.get("biological_processes", [])
        )
        insights["disease_relevant_processes"] = disease_relevant_processes
        
        # Assess therapeutic potential based on critical and disease-relevant processes
        critical_count = len(critical_processes)
        disease_count = len(disease_relevant_processes)
        
        if critical_count >= 2 and disease_count >= 1:
            insights["therapeutic_potential"] = "High - Multiple critical processes with disease relevance"
        elif critical_count >= 1 or disease_count >= 2:
            insights["therapeutic_potential"] = "Medium - Important processes identified"
        else:
            insights["therapeutic_potential"] = "Low - Limited critical or disease-relevant processes"
        
        # Generate research priorities based on actual processes
        research_priorities = []
        if critical_count == 0:
            research_priorities.append("Investigate potential critical biological functions")
        if disease_count == 0:
            research_priorities.append("Explore disease associations and pathological roles")
        if enhanced_analysis.get("process_complexity_score", 0) < 0.3:
            research_priorities.append("Characterize additional biological functions")
        
        insights["research_priorities"] = research_priorities
        
        # Confidence assessment based on data quality
        total_processes = basic_result.get("total_processes", 0)
        confidence_factors = []
        if total_processes >= 10:
            confidence_factors.append("Comprehensive process data available")
        if critical_count > 0:
            confidence_factors.append(f"{critical_count} critical processes identified")
        if disease_count > 0:
            confidence_factors.append(f"{disease_count} disease-relevant processes found")
        if enhanced_analysis.get("process_complexity_score", 0) > 0.5:
            confidence_factors.append("High process complexity indicates well-characterized gene")
        
        if len(confidence_factors) >= 2:
            insights["confidence_assessment"] = "High confidence - " + "; ".join(confidence_factors)
        elif len(confidence_factors) == 1:
            insights["confidence_assessment"] = "Medium confidence - " + confidence_factors[0]
        else:
            insights["confidence_assessment"] = "Low confidence - Limited process data available"
        
        return insights

    def _identify_disease_relevant_processes(self, processes: List[str]) -> List[Dict[str, Any]]:
        """Identify disease-relevant biological processes"""
        disease_relevant = []
        
        # Keywords that indicate disease relevance
        disease_keywords = {
            "cancer": ["cancer", "tumor", "oncogenic", "carcinogenesis", "metastasis"],
            "apoptosis": ["apoptosis", "programmed cell death", "cell death"],
            "dna_damage": ["dna damage", "dna repair", "genomic instability"],
            "inflammation": ["inflammation", "inflammatory", "immune response"],
            "metabolism": ["metabolic disorder", "metabolism", "biosynthesis"],
            "signaling": ["signal transduction", "signaling pathway", "receptor"],
            "stress": ["stress response", "oxidative stress", "heat shock"],
            "development": ["developmental disorder", "differentiation", "morphogenesis"]
        }
        
        for process in processes:
            process_lower = process.lower()
            matching_categories = []
            
            for category, keywords in disease_keywords.items():
                for keyword in keywords:
                    if keyword.lower() in process_lower:
                        matching_categories.append({
                            "category": category,
                            "keyword": keyword,
                            "relevance_level": self._assess_disease_relevance_level(category)
                        })
                        break
            
            if matching_categories:
                # Sort by relevance level
                matching_categories.sort(key=lambda x: x["relevance_level"], reverse=True)
                disease_relevant.append({
                    "process": process,
                    "disease_categories": matching_categories,
                    "primary_category": matching_categories[0]["category"],
                    "relevance_level": matching_categories[0]["relevance_level"]
                })
        
        # Sort by relevance level
        disease_relevant.sort(key=lambda x: x["relevance_level"], reverse=True)
        return disease_relevant

    def _assess_disease_relevance_level(self, category: str) -> str:
        """Assess the disease relevance level of a process category"""
        high_relevance = ["cancer", "apoptosis", "dna_damage"]
        medium_relevance = ["inflammation", "signaling", "metabolism"]
        low_relevance = ["stress", "development"]
        
        if category in high_relevance:
            return "high"
        elif category in medium_relevance:
            return "medium"
        else:
            return "low"

    def _calculate_category_confidence(self, process: str, keyword: str) -> float:
        """Calculate confidence score for process categorization"""
        process_lower = process.lower()
        keyword_lower = keyword.lower()
        
        # Exact match gets highest confidence
        if keyword_lower == process_lower:
            return 1.0
        
        # Keyword at start of process gets high confidence
        if process_lower.startswith(keyword_lower):
            return 0.9
        
        # Keyword in process gets medium confidence
        if keyword_lower in process_lower:
            return 0.7
        
        # Partial match gets lower confidence
        return 0.3

    def _assess_disease_relevance(self, critical_type: str) -> str:
        """Assess disease relevance of critical process types"""
        high_relevance = ["apoptosis", "cell cycle", "dna repair"]
        medium_relevance = ["transcription regulation", "signal transduction"]
        
        if critical_type in high_relevance:
            return "high"
        elif critical_type in medium_relevance:
            return "medium"
        else:
            return "low"

    def _calculate_complexity_score(self, processes: List[str]) -> float:
        """Calculate complexity score based on number and diversity of processes"""
        if not processes:
            return 0.0
        
        # Base score from number of processes (normalized to 0-1)
        base_score = min(len(processes) / 20.0, 1.0)
        
        # Diversity score based on unique keywords
        unique_keywords = set()
        for process in processes:
            words = process.lower().split()
            unique_keywords.update(words)
        
        diversity_score = min(len(unique_keywords) / 50.0, 1.0)
        
        # Combined score
        return (base_score + diversity_score) / 2.0

    def _calculate_functional_similarity(self, processes1: List[str], processes2: List[str]) -> float:
        """Calculate functional similarity between two sets of biological processes"""
        if not processes1 or not processes2:
            return 0.0
        
        # Convert to sets of lowercase words for comparison
        words1 = set()
        for process in processes1:
            words1.update(process.lower().split())
        
        words2 = set()
        for process in processes2:
            words2.update(process.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0

    def _find_shared_processes(self, processes1: List[str], processes2: List[str]) -> List[str]:
        """Find shared biological processes between two gene sets"""
        shared = []
        
        for process1 in processes1:
            for process2 in processes2:
                # Simple similarity check
                if (process1.lower() in process2.lower() or 
                    process2.lower() in process1.lower() or
                    any(word in process2.lower() for word in process1.lower().split())):
                    shared.append(process1)
                    break
        
        return shared

    def _calculate_confidence_level(self, basic_result: Dict[str, Any]) -> str:
        """Calculate overall confidence level of the analysis"""
        total_processes = basic_result.get("total_processes", 0)
        target_processes = len(basic_result.get("target_processes_found", []))
        
        if total_processes >= 15 and target_processes >= 3:
            return "high"
        elif total_processes >= 8 and target_processes >= 1:
            return "medium"
        else:
            return "low"

    def _get_timestamp(self) -> str:
        """Get current timestamp for analysis metadata"""
        from datetime import datetime
        return datetime.now().isoformat()


class HPAGetEnhancedComparativeExpressionTool(HPASearchApiTool):
    """
    Enhanced tool for comparing gene expression levels between cell lines and healthy tissues.
    Leverages HPAGetComparativeExpressionTool with additional features including detailed expression analysis,
    statistical significance assessment, expression level categorization, and comprehensive comparison insights.
    """
    def __init__(self, tool_config):
        super().__init__(tool_config)
        
        # Enhanced cell line mapping with additional metadata
        self.cell_line_data = {
            "ishikawa": {
                "column": "cell_RNA_ishikawa_heraklio",
                "type": "endometrial adenocarcinoma",
                "origin": "endometrium",
                "description": "Human endometrial adenocarcinoma cell line"
            },
            "hela": {
                "column": "cell_RNA_hela",
                "type": "cervical adenocarcinoma",
                "origin": "cervix",
                "description": "Human cervical adenocarcinoma cell line"
            },
            "mcf7": {
                "column": "cell_RNA_mcf7",
                "type": "breast adenocarcinoma",
                "origin": "breast",
                "description": "Human breast adenocarcinoma cell line"
            },
            "a549": {
                "column": "cell_RNA_a549",
                "type": "lung adenocarcinoma",
                "origin": "lung",
                "description": "Human lung adenocarcinoma cell line"
            },
            "hepg2": {
                "column": "cell_RNA_hepg2",
                "type": "hepatocellular carcinoma",
                "origin": "liver",
                "description": "Human hepatocellular carcinoma cell line"
            },
            "jurkat": {
                "column": "cell_RNA_jurkat",
                "type": "acute T cell leukemia",
                "origin": "blood",
                "description": "Human acute T cell leukemia cell line"
            },
            "pc3": {
                "column": "cell_RNA_pc3",
                "type": "prostate adenocarcinoma",
                "origin": "prostate",
                "description": "Human prostate adenocarcinoma cell line"
            },
            "rh30": {
                "column": "cell_RNA_rh30",
                "type": "rhabdomyosarcoma",
                "origin": "muscle",
                "description": "Human rhabdomyosarcoma cell line"
            },
            "siha": {
                "column": "cell_RNA_siha",
                "type": "cervical squamous cell carcinoma",
                "origin": "cervix",
                "description": "Human cervical squamous cell carcinoma cell line"
            },
            "u251": {
                "column": "cell_RNA_u251",
                "type": "glioblastoma",
                "origin": "brain",
                "description": "Human glioblastoma cell line"
            }
        }
        
        # Expression level categories
        self.expression_categories = {
            "very_high": {"min": 50.0, "description": "Very high expression"},
            "high": {"min": 10.0, "max": 49.99, "description": "High expression"},
            "medium": {"min": 1.0, "max": 9.99, "description": "Medium expression"},
            "low": {"min": 0.1, "max": 0.99, "description": "Low expression"},
            "very_low": {"max": 0.099, "description": "Very low expression"}
        }

    def run(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced comparative expression analysis with detailed insights.
        
        Args:
            gene_name (str): Gene name or symbol (e.g., 'TP53', 'BRCA1', 'EGFR')
            cell_line (str): Cell line name from supported list
            include_statistical_analysis (bool): Whether to include statistical significance analysis
            include_expression_breakdown (bool): Whether to include detailed expression breakdown
            include_therapeutic_insights (bool): Whether to include therapeutic relevance insights
        
        Returns:
            Dict containing comprehensive comparative expression analysis
        """
        gene_name = arguments.get("gene_name")
        cell_line = (arguments.get("cell_line") or "").lower()
        include_statistical_analysis = arguments.get("include_statistical_analysis", True)
        include_expression_breakdown = arguments.get("include_expression_breakdown", True)
        include_therapeutic_insights = arguments.get("include_therapeutic_insights", True)
        
        if not gene_name:
            return {"error": "Parameter 'gene_name' is required"}
        if not cell_line:
            return {"error": "Parameter 'cell_line' is required"}
        
        # Validate cell line with enhanced recommendations
        cell_line_info = self.cell_line_data.get(cell_line)
        if not cell_line_info:
            available_lines = list(self.cell_line_data.keys())
            
            # Find similar cell line names
            similar_lines = []
            for valid_line in available_lines:
                if cell_line in valid_line or valid_line in cell_line:
                    similar_lines.append(valid_line)
            
            error_msg = f"Unsupported cell_line '{cell_line}'. "
            if similar_lines:
                error_msg += f"Similar options: {similar_lines}. "
            error_msg += f"All supported cell lines: {available_lines}"
            return {"error": error_msg}
        
        try:
            # Use the base comparative expression tool
            base_tool = HPAGetComparativeExpressionTool({})
            base_result = base_tool.run({
                "gene_name": gene_name,
                "cell_line": cell_line
            })
            
            if "error" in base_result:
                return base_result
            
            # Enhance the result with additional analysis
            enhanced_result = self._enhance_comparative_analysis(
                base_result, 
                cell_line_info,
                include_statistical_analysis,
                include_expression_breakdown,
                include_therapeutic_insights
            )
            
            return enhanced_result
            
        except Exception as e:
            return {
                "error": f"Failed to perform enhanced comparative expression analysis: {str(e)}",
                "gene_name": gene_name,
                "cell_line": cell_line
            }

    def _enhance_comparative_analysis(self, base_result: Dict[str, Any], 
                                    cell_line_info: Dict[str, Any],
                                    include_statistical_analysis: bool,
                                    include_expression_breakdown: bool,
                                    include_therapeutic_insights: bool) -> Dict[str, Any]:
        """Enhance base comparative analysis with additional insights"""
        
        # Extract expression values
        cell_expr = base_result.get("cell_line_expression", "N/A")
        tissue_expr = base_result.get("healthy_tissue_expression", "N/A")
        
        enhanced_result = {
            # Basic information
            "gene_name": base_result.get("gene_name"),
            "gene_symbol": base_result.get("gene_symbol"),
            "gene_synonym": base_result.get("gene_synonym"),
            "cell_line": base_result.get("cell_line"),
            "cell_line_info": cell_line_info,
            
            # Expression data
            "cell_line_expression": cell_expr,
            "healthy_tissue_expression": tissue_expr,
            "expression_unit": base_result.get("expression_unit"),
            
            # Enhanced analysis
            "expression_analysis": {},
            "comparison_analysis": {},
            "therapeutic_insights": {},
            "metadata": {}
        }
        
        # Expression breakdown analysis
        if include_expression_breakdown:
            enhanced_result["expression_analysis"] = self._analyze_expression_levels(cell_expr, tissue_expr)
        
        # Statistical analysis
        if include_statistical_analysis:
            enhanced_result["comparison_analysis"] = self._perform_statistical_analysis(cell_expr, tissue_expr)
        
        # Therapeutic insights
        if include_therapeutic_insights:
            enhanced_result["therapeutic_insights"] = self._generate_therapeutic_insights(
                cell_expr, tissue_expr, cell_line_info, base_result.get("gene_symbol")
            )
        
        # Enhanced comparison summary
        enhanced_result["comparison_summary"] = self._generate_enhanced_comparison_summary(
            cell_expr, tissue_expr, cell_line_info
        )
        
        # Metadata
        enhanced_result["metadata"] = {
            "analysis_timestamp": self._get_timestamp(),
            "analysis_version": "2.0",
            "data_source": "Human Protein Atlas",
            "confidence_level": self._calculate_confidence_level(cell_expr, tissue_expr)
        }
        
        return enhanced_result

    def _analyze_expression_levels(self, cell_expr: str, tissue_expr: str) -> Dict[str, Any]:
        """Analyze and categorize expression levels"""
        analysis = {
            "cell_line_category": "unknown",
            "tissue_category": "unknown",
            "expression_difference": "unknown",
            "fold_change": "N/A",
            "expression_ratio": "N/A"
        }
        
        try:
            cell_val = float(cell_expr) if cell_expr and cell_expr != "N/A" else None
            tissue_val = float(tissue_expr) if tissue_expr and tissue_expr != "N/A" else None
            
            if cell_val is not None:
                analysis["cell_line_category"] = self._categorize_expression_level(cell_val)
            
            if tissue_val is not None:
                analysis["tissue_category"] = self._categorize_expression_level(tissue_val)
            
            if cell_val is not None and tissue_val is not None and tissue_val > 0:
                fold_change = cell_val / tissue_val
                analysis["fold_change"] = fold_change
                analysis["expression_ratio"] = f"{fold_change:.2f}"
                
                if fold_change > 2:
                    analysis["expression_difference"] = "significantly higher in cell line"
                elif fold_change < 0.5:
                    analysis["expression_difference"] = "significantly higher in healthy tissues"
                else:
                    analysis["expression_difference"] = "similar expression levels"
            
        except (ValueError, TypeError):
            pass
        
        return analysis

    def _categorize_expression_level(self, expression_value: float) -> str:
        """Categorize expression level based on nTPM value"""
        for category, criteria in self.expression_categories.items():
            min_val = criteria.get("min", 0)
            max_val = criteria.get("max", float('inf'))
            
            if min_val <= expression_value <= max_val:
                return category
        
        return "unknown"

    def _perform_statistical_analysis(self, cell_expr: str, tissue_expr: str) -> Dict[str, Any]:
        """Perform statistical analysis of expression differences"""
        analysis = {
            "statistical_significance": "unknown",
            "effect_size": "unknown",
            "confidence_level": "unknown",
            "interpretation": "Insufficient data for statistical analysis"
        }
        
        try:
            cell_val = float(cell_expr) if cell_expr and cell_expr != "N/A" else None
            tissue_val = float(tissue_expr) if tissue_expr and tissue_expr != "N/A" else None
            
            if cell_val is not None and tissue_val is not None:
                # Calculate fold change
                fold_change = cell_val / tissue_val if tissue_val > 0 else float('inf')
                
                # Determine statistical significance based on fold change
                if fold_change > 5 or fold_change < 0.2:
                    analysis["statistical_significance"] = "high"
                    analysis["confidence_level"] = "high"
                    analysis["interpretation"] = "Strong evidence of differential expression"
                elif fold_change > 2 or fold_change < 0.5:
                    analysis["statistical_significance"] = "medium"
                    analysis["confidence_level"] = "medium"
                    analysis["interpretation"] = "Moderate evidence of differential expression"
                else:
                    analysis["statistical_significance"] = "low"
                    analysis["confidence_level"] = "low"
                    analysis["interpretation"] = "Limited evidence of differential expression"
                
                # Effect size assessment
                if abs(fold_change - 1) > 3:
                    analysis["effect_size"] = "large"
                elif abs(fold_change - 1) > 1:
                    analysis["effect_size"] = "medium"
                else:
                    analysis["effect_size"] = "small"
            
        except (ValueError, TypeError):
            pass
        
        return analysis

    def _generate_therapeutic_insights(self, cell_expr: str, tissue_expr: str, 
                                     cell_line_info: Dict[str, Any], gene_symbol: str) -> Dict[str, Any]:
        """Generate therapeutic insights based on expression patterns"""
        insights = {
            "therapeutic_potential": "unknown",
            "targeting_strategy": "unknown",
            "biomarker_potential": "unknown",
            "clinical_relevance": "unknown",
            "recommendations": []
        }
        
        try:
            cell_val = float(cell_expr) if cell_expr and cell_expr != "N/A" else None
            tissue_val = float(tissue_expr) if tissue_expr and tissue_expr != "N/A" else None
            
            if cell_val is not None and tissue_val is not None:
                fold_change = cell_val / tissue_val if tissue_val > 0 else float('inf')
                
                # Assess therapeutic potential
                if fold_change > 3:
                    insights["therapeutic_potential"] = "high"
                    insights["targeting_strategy"] = "direct targeting"
                    insights["biomarker_potential"] = "high"
                    insights["clinical_relevance"] = "strong candidate for therapeutic intervention"
                    insights["recommendations"].append("Consider as primary therapeutic target")
                    insights["recommendations"].append("High potential for biomarker development")
                elif fold_change > 2:
                    insights["therapeutic_potential"] = "medium"
                    insights["targeting_strategy"] = "selective targeting"
                    insights["biomarker_potential"] = "medium"
                    insights["clinical_relevance"] = "moderate candidate for therapeutic intervention"
                    insights["recommendations"].append("Consider in combination therapy approaches")
                elif fold_change < 0.5:
                    insights["therapeutic_potential"] = "low"
                    insights["targeting_strategy"] = "not recommended"
                    insights["biomarker_potential"] = "low"
                    insights["clinical_relevance"] = "limited therapeutic potential"
                    insights["recommendations"].append("Focus on alternative targets")
                else:
                    insights["therapeutic_potential"] = "low"
                    insights["targeting_strategy"] = "context-dependent"
                    insights["biomarker_potential"] = "low"
                    insights["clinical_relevance"] = "requires additional validation"
                    insights["recommendations"].append("Further investigation needed")
                
                # Add cell line specific insights
                cancer_type = cell_line_info.get("type", "")
                if cancer_type:
                    insights["recommendations"].append(f"Relevant for {cancer_type} research")
                
        except (ValueError, TypeError):
            pass
        
        return insights

    def _generate_enhanced_comparison_summary(self, cell_expr: str, tissue_expr: str, 
                                            cell_line_info: Dict[str, Any]) -> str:
        """Generate enhanced comparison summary with detailed insights"""
        try:
            cell_val = float(cell_expr) if cell_expr and cell_expr != "N/A" else None
            tissue_val = float(tissue_expr) if tissue_expr and tissue_expr != "N/A" else None
            
            if cell_val is None or tissue_val is None:
                return "Insufficient data for detailed comparison"
            
            fold_change = cell_val / tissue_val if tissue_val > 0 else float('inf')
            cancer_type = cell_line_info.get("type", "cancer")
            
            if fold_change > 5:
                return f"Expression is dramatically higher in {cancer_type} cell line ({cell_val:.2f} nTPM) compared to healthy tissues ({tissue_val:.2f} nTPM), representing a {fold_change:.1f}-fold increase. This suggests strong oncogenic potential and high therapeutic targeting potential."
            elif fold_change > 2:
                return f"Expression is significantly higher in {cancer_type} cell line ({cell_val:.2f} nTPM) compared to healthy tissues ({tissue_val:.2f} nTPM), representing a {fold_change:.1f}-fold increase. This indicates potential oncogenic role and moderate therapeutic potential."
            elif fold_change < 0.5:
                return f"Expression is significantly lower in {cancer_type} cell line ({cell_val:.2f} nTPM) compared to healthy tissues ({tissue_val:.2f} nTPM), representing a {1/fold_change:.1f}-fold decrease. This suggests potential tumor suppressor function or loss of expression in cancer."
            else:
                return f"Expression levels are similar between {cancer_type} cell line ({cell_val:.2f} nTPM) and healthy tissues ({tissue_val:.2f} nTPM), with a {fold_change:.1f}-fold ratio. This indicates stable expression across conditions."
                
        except (ValueError, TypeError):
            return "Failed to calculate detailed expression comparison"

    def _calculate_confidence_level(self, cell_expr: str, tissue_expr: str) -> str:
        """Calculate confidence level of the analysis"""
        try:
            cell_val = float(cell_expr) if cell_expr and cell_expr != "N/A" else None
            tissue_val = float(tissue_expr) if tissue_expr and tissue_expr != "N/A" else None
            
            if cell_val is not None and tissue_val is not None:
                return "high"
            elif cell_val is not None or tissue_val is not None:
                return "medium"
            else:
                return "low"
        except (ValueError, TypeError):
            return "low"

    def _get_timestamp(self) -> str:
        """Get current timestamp for analysis metadata"""
        from datetime import datetime
        return datetime.now().isoformat()


def test_enhanced_comparative_expression(test_genes: list, test_cell_lines: list):
    """Test HPAGetEnhancedComparativeExpressionTool with various scenarios"""
    
    tool = HPAGetEnhancedComparativeExpressionTool({})
    
    # Test Case 1: Basic functionality
    print("\n📋 Test Case 1: Basic Enhanced Comparative Analysis")
    print("-" * 50)
    
    for gene in test_genes[:2]:  # Test first 2 genes
        for cell_line in test_cell_lines[:2]:  # Test first 2 cell lines
            print(f"\nTesting {gene} in {cell_line} cell line:")
            
            result = tool.run({
                "gene_name": gene,
                "cell_line": cell_line,
                "include_statistical_analysis": True,
                "include_expression_breakdown": True,
                "include_therapeutic_insights": True
            })
            
            if "error" not in result:
                print(f"  ✅ Success: {result['gene_symbol']}")
                print(f"  📊 Expression: {result['cell_line_expression']} vs {result['healthy_tissue_expression']} nTPM")
                print(f"  📈 Fold change: {result['expression_analysis']['expression_ratio']}")
                print(f"  🎯 Therapeutic potential: {result['therapeutic_insights']['therapeutic_potential']}")
                print(f"  📝 Summary: {result['comparison_summary'][:100]}...")
            else:
                print(f"  ❌ Error: {result['error']}")
    
    # Test Case 2: Statistical analysis focus
    print("\n📋 Test Case 2: Statistical Analysis Focus")
    print("-" * 50)
    
    result = tool.run({
        "gene_name": "TP53",
        "cell_line": "hela",
        "include_statistical_analysis": True,
        "include_expression_breakdown": True,
        "include_therapeutic_insights": False
    })
    
    if "error" not in result:
        print(f"✅ Statistical analysis for TP53 in Hela:")
        comparison = result['comparison_analysis']
        print(f"  📊 Significance: {comparison['statistical_significance']}")
        print(f"  📈 Effect size: {comparison['effect_size']}")
        print(f"  🎯 Confidence: {comparison['confidence_level']}")
        print(f"  📝 Interpretation: {comparison['interpretation']}")
        
        expression = result['expression_analysis']
        print(f"  📊 Cell line category: {expression['cell_line_category']}")
        print(f"  📊 Tissue category: {expression['tissue_category']}")
        print(f"  📈 Expression difference: {expression['expression_difference']}")
    else:
        print(f"❌ Error: {result['error']}")
    
    # Test Case 3: Therapeutic insights focus
    print("\n📋 Test Case 3: Therapeutic Insights Focus")
    print("-" * 50)
    
    result = tool.run({
        "gene_name": "EGFR",
        "cell_line": "a549",
        "include_statistical_analysis": True,
        "include_expression_breakdown": True,
        "include_therapeutic_insights": True
    })
    
    if "error" not in result:
        print(f"✅ Therapeutic insights for EGFR in A549:")
        insights = result['therapeutic_insights']
        print(f"  🎯 Therapeutic potential: {insights['therapeutic_potential']}")
        print(f"  🎯 Targeting strategy: {insights['targeting_strategy']}")
        print(f"  🎯 Biomarker potential: {insights['biomarker_potential']}")
        print(f"  🎯 Clinical relevance: {insights['clinical_relevance']}")
        print(f"  📝 Recommendations:")
        for rec in insights['recommendations']:
            print(f"    - {rec}")
    else:
        print(f"❌ Error: {result['error']}")
    
    # Test Case 4: Cell line metadata validation
    print("\n📋 Test Case 4: Cell Line Metadata Validation")
    print("-" * 50)
    
    for cell_line in test_cell_lines:
        result = tool.run({
            "gene_name": "TP53",
            "cell_line": cell_line,
            "include_statistical_analysis": False,
            "include_expression_breakdown": False,
            "include_therapeutic_insights": False
        })
        
        if "error" not in result:
            cell_info = result['cell_line_info']
            print(f"✅ {cell_line}: {cell_info['type']} (origin: {cell_info['origin']})")
            print(f"   Description: {cell_info['description']}")
        else:
            print(f"❌ {cell_line}: {result['error']}")


def test_comprehensive_biological_process(test_genes: list):
    """Test HPAGetComprehensiveBiologicalProcessTool with various scenarios"""
    
    tool = HPAGetComprehensiveBiologicalProcessTool({})
    
    # Test Case 1: Basic comprehensive analysis
    print("\n📋 Test Case 1: Basic Comprehensive Biological Process Analysis")
    print("-" * 60)
    
    for gene in test_genes[:2]:  # Test first 2 genes
        print(f"\nTesting {gene}:")
        
        result = tool.run({
            "gene_name": gene,
            "include_categorization": True,
            "include_pathway_analysis": True,
            "include_comparative_analysis": False,
            "max_processes": 30,
            "filter_critical_only": False
        })
        
        if "error" not in result:
            print(f"  ✅ Success: {result['gene_symbol']}")
            print(f"  📊 Total processes: {result['analysis_summary']['total_processes']}")
            print(f"  🎯 Critical processes: {result['analysis_summary']['critical_processes_found']}")
            print(f"  📈 Categorized processes: {result['analysis_summary']['categorized_processes']}")
            print(f"  🛤️ Pathway involvement: {result['analysis_summary']['pathway_involvement']}")
            
            insights = result['functional_insights']
            print(f"  🎯 Critical processes found: {len(insights['critical_processes'])}")
            print(f"  🎯 Disease-relevant processes: {len(insights['disease_relevant_processes'])}")
            print(f"  🎯 Therapeutic potential: {insights['therapeutic_potential']}")
            
            # Show critical processes
            if insights['critical_processes']:
                print(f"  🔬 Critical processes:")
                for cp in insights['critical_processes'][:3]:  # Show first 3
                    print(f"    - {cp['process']} ({cp['critical_type']})")
            
            # Show disease-relevant processes
            if insights['disease_relevant_processes']:
                print(f"  🏥 Disease-relevant processes:")
                for dp in insights['disease_relevant_processes'][:3]:  # Show first 3
                    print(f"    - {dp['process']} ({dp['primary_category']}, {dp['relevance_level']} relevance)")
        else:
            print(f"  ❌ Error: {result['error']}")
    
    # Test Case 2: Critical processes focus
    print("\n📋 Test Case 2: Critical Processes Focus")
    print("-" * 60)
    
    result = tool.run({
        "gene_name": "TP53",
        "include_categorization": True,
        "include_pathway_analysis": True,
        "include_comparative_analysis": False,
        "max_processes": 20,
        "filter_critical_only": True
    })
    
    if "error" not in result:
        print(f"✅ Critical processes analysis for TP53:")
        critical_processes = result['enhanced_analysis']['critical_processes']
        if critical_processes:
            print(f"  🎯 Found {len(critical_processes)} critical processes:")
            for cp in critical_processes:
                print(f"    - {cp['process']} ({cp['critical_type']}) - {cp['disease_relevance']} relevance")
        else:
            print("  📝 No critical processes found")
        
        categorized = result['enhanced_analysis']['categorized_processes']
        print(f"  📊 Process categories: {list(categorized.keys())}")
    else:
        print(f"❌ Error: {result['error']}")
    
    # Test Case 3: Pathway analysis focus
    print("\n📋 Test Case 3: Pathway Analysis Focus")
    print("-" * 60)
    
    result = tool.run({
        "gene_name": "BRCA1",
        "include_categorization": True,
        "include_pathway_analysis": True,
        "include_comparative_analysis": False,
        "max_processes": 50,
        "filter_critical_only": False
    })
    
    if "error" not in result:
        print(f"✅ Pathway analysis for BRCA1:")
        pathway_analysis = result['enhanced_analysis']['pathway_analysis']
        if pathway_analysis:
            print(f"  🛤️ Found {len(pathway_analysis)} pathway involvements:")
            for pathway, data in pathway_analysis.items():
                print(f"    - {pathway}: score {data['involvement_score']}, confidence {data['pathway_confidence']:.2f}")
        else:
            print("  📝 No pathway involvements found")
        
        complexity_score = result['enhanced_analysis']['process_complexity_score']
        print(f"  📊 Process complexity score: {complexity_score:.2f}")
    else:
        print(f"❌ Error: {result['error']}")
    
    # Test Case 4: Full analysis with comparative data
    print("\n📋 Test Case 4: Full Analysis with Comparative Data")
    print("-" * 60)
    
    result = tool.run({
        "gene_name": "EGFR",
        "include_categorization": True,
        "include_pathway_analysis": True,
        "include_comparative_analysis": True,
        "max_processes": 40,
        "filter_critical_only": False
    })
    
    if "error" not in result:
        print(f"✅ Full analysis for EGFR:")
        
        # Show research priorities and confidence
        insights = result['functional_insights']
        print(f"  📝 Research priorities: {', '.join(insights['research_priorities'])}")
        print(f"  🎯 Confidence assessment: {insights['confidence_assessment']}")
        
        # Show detailed critical and disease-relevant processes
        if insights['critical_processes']:
            print(f"  🔬 Critical processes ({len(insights['critical_processes'])} total):")
            for cp in insights['critical_processes']:
                print(f"    - {cp['process']} ({cp['critical_type']}, {cp['disease_relevance']} disease relevance)")
        
        if insights['disease_relevant_processes']:
            print(f"  🏥 Disease-relevant processes ({len(insights['disease_relevant_processes'])} total):")
            for dp in insights['disease_relevant_processes']:
                print(f"    - {dp['process']} ({dp['primary_category']}, {dp['relevance_level']} relevance)")
        
        # Show comparative analysis
        comparative = result['comparative_analysis']
        if 'related_genes' in comparative and comparative['related_genes']:
            print(f"  🔗 Related genes: {len(comparative['related_genes'])} found")
            for rg in comparative['related_genes'][:3]:  # Show first 3
                print(f"    - {rg['gene_name']} (similarity: {rg['functional_similarity']:.2f})")
        else:
            print("  📝 No related genes found")
        
        # Show metadata
        metadata = result['metadata']
        print(f"  📊 Analysis version: {metadata['analysis_version']}")
        print(f"  📊 Confidence level: {metadata['confidence_level']}")
        print(f"  📊 Data source: {metadata['data_source']}")
    else:
        print(f"❌ Error: {result['error']}")


def test_error_handling():
    """Test error handling and edge cases for both tools"""
    
    print("\n📋 Test Case 1: Invalid Gene Names")
    print("-" * 40)
    
    # Test with invalid gene names
    enhanced_tool = HPAGetEnhancedComparativeExpressionTool({})
    comprehensive_tool = HPAGetComprehensiveBiologicalProcessTool({})
    
    invalid_genes = ["INVALID_GENE_123", "NONEXISTENT_GENE", "GENE_WITH_SPECIAL_CHARS_!@#"]
    
    for gene in invalid_genes:
        print(f"\nTesting invalid gene: {gene}")
        
        # Test enhanced comparative expression
        result1 = enhanced_tool.run({
            "gene_name": gene,
            "cell_line": "hela"
        })
        if "error" in result1:
            print(f"  ✅ Enhanced tool: {result1['error']}")
        else:
            print(f"  ⚠️ Enhanced tool: Unexpected success")
        
        # Test comprehensive biological process
        result2 = comprehensive_tool.run({
            "gene_name": gene
        })
        if "error" in result2:
            print(f"  ✅ Comprehensive tool: {result2['error']}")
        else:
            print(f"  ⚠️ Comprehensive tool: Unexpected success")
    
    print("\n📋 Test Case 2: Invalid Cell Lines")
    print("-" * 40)
    
    invalid_cell_lines = ["invalid_cell", "cancer_cell", "test_cell_line"]
    
    for cell_line in invalid_cell_lines:
        print(f"\nTesting invalid cell line: {cell_line}")
        
        result = enhanced_tool.run({
            "gene_name": "TP53",
            "cell_line": cell_line
        })
        if "error" in result:
            print(f"  ✅ Error handling: {result['error']}")
        else:
            print(f"  ⚠️ Unexpected success")
    
    print("\n📋 Test Case 3: Missing Parameters")
    print("-" * 40)
    
    # Test missing gene_name
    result1 = enhanced_tool.run({
        "cell_line": "hela"
    })
    if "error" in result1:
        print(f"  ✅ Missing gene_name: {result1['error']}")
    else:
        print(f"  ⚠️ Missing gene_name: Unexpected success")
    
    # Test missing cell_line
    result2 = enhanced_tool.run({
        "gene_name": "TP53"
    })
    if "error" in result2:
        print(f"  ✅ Missing cell_line: {result2['error']}")
    else:
        print(f"  ⚠️ Missing cell_line: Unexpected success")
    
    # Test missing gene_name for comprehensive tool
    result3 = comprehensive_tool.run({})
    if "error" in result3:
        print(f"  ✅ Missing gene_name (comprehensive): {result3['error']}")
    else:
        print(f"  ⚠️ Missing gene_name (comprehensive): Unexpected success")
    
    print("\n📋 Test Case 4: Edge Cases")
    print("-" * 40)
    
    # Test with empty string
    result1 = enhanced_tool.run({
        "gene_name": "",
        "cell_line": "hela"
    })
    if "error" in result1:
        print(f"  ✅ Empty gene_name: {result1['error']}")
    else:
        print(f"  ⚠️ Empty gene_name: Unexpected success")
    
    # Test with whitespace-only
    result2 = enhanced_tool.run({
        "gene_name": "   ",
        "cell_line": "hela"
    })
    if "error" in result2:
        print(f"  ✅ Whitespace gene_name: {result2['error']}")
    else:
        print(f"  ⚠️ Whitespace gene_name: Unexpected success")
    
    # Test with very long gene name
    long_gene = "A" * 1000
    result3 = enhanced_tool.run({
        "gene_name": long_gene,
        "cell_line": "hela"
    })
    if "error" in result3:
        print(f"  ✅ Very long gene_name: {result3['error']}")
    else:
        print(f"  ⚠️ Very long gene_name: Unexpected success")


#### HumanBase Tools ####

@dataclass
class HumanBaseResult:
    """Standardized result structure for HumanBase analysis"""
    tissue: str
    genes: List[str]
    interaction_summary: str
    network_strength: str  # high/medium/low/minimal
    key_interactions: List[Dict[str, Any]]
    biological_processes: List[str]
    tissue_specificity: str  # high/medium/low
    clinical_relevance: str

class BaseHumanBaseTool(ABC):
    """Abstract base class for HumanBase tissue-specific analysis tools"""
    
    def __init__(self):
        self.base_url = "https://hb.flatironinstitute.org/api"
        self.max_retries = 3
        self.retry_delay = 1.0
        self._gene_cache = {}
        # Valid tissue types in HumanBase
        self._valid_tissues = {
            'adipose-tissue', 'adrenal-gland', 'blood', 'bone', 'brain', 'breast', 
            'colon', 'endothelial-cell', 'esophagus', 'heart', 'kidney', 'liver', 
            'lung', 'muscle', 'ovary', 'pancreas', 'prostate', 'skin', 'stomach', 
            'testis', 'thyroid', 'uterus', 'artery-endothelial-cell', 
            'gut-endothelial-cell', 'vein-endothelial-cell'
        }
    
    @abstractmethod
    def get_interaction_type(self) -> str:
        """Return the specific interaction type for this tool"""
        pass
    
    @abstractmethod
    def interpret_interactions(self, interactions: List[Dict], tissue: str) -> str:
        """Interpret the biological significance of interactions"""
        pass
    
    def _normalize_tissue(self, tissue: str) -> str:
        """Normalize tissue name for API"""
        return tissue.replace(" ", "-").replace("_", "-").lower()
    
    def _validate_tissue(self, tissue: str) -> Tuple[bool, str, List[str]]:
        """
        Validate tissue name and suggest alternatives if invalid
        Returns: (is_valid, normalized_tissue, suggestions)
        """
        normalized = self._normalize_tissue(tissue)
        
        if normalized in self._valid_tissues:
            return True, normalized, []
        
        # Find close matches for suggestions
        suggestions = []
        if 'endothelial' in tissue.lower():
            suggestions = ['endothelial-cell', 'artery-endothelial-cell', 'gut-endothelial-cell', 'vein-endothelial-cell']
        else:
            # Find tissues that contain part of the input
            suggestions = [t for t in self._valid_tissues if any(part in t for part in normalized.split('-'))]
            suggestions = suggestions[:3]  # Limit to 3 suggestions
        
        return False, normalized, suggestions
    
    def _get_entrez_ids(self, gene_names: List[str]) -> List[str]:
        """Convert gene names to Entrez IDs with caching"""
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        entrez_ids = []
        
        # Resolve gene names to official symbols first
        resolved_genes = []
        for gene in gene_names:
            if gene in self._gene_cache:
                resolved_genes.append(self._gene_cache[gene])
            else:
                official_name = get_official_gene_name(gene)
                self._gene_cache[gene] = official_name
                resolved_genes.append(official_name)
                print(f"[HUMANBASE] INFO: Resolved {gene} → {official_name}", flush=True)
        
        for gene in resolved_genes:
            cache_key = f"entrez_{gene}"
            if cache_key in self._gene_cache:
                entrez_ids.append(self._gene_cache[cache_key])
                continue
                
            params = {
                'db': 'gene',
                'term': f"{gene}[gene] AND Homo sapiens[orgn]",
                'retmode': 'xml',
                'retmax': '1'
            }
            
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                xml_data = response.text
                
                start_idx = xml_data.find("<Id>")
                end_idx = xml_data.find("</Id>")
                
                if start_idx != -1 and end_idx != -1:
                    entrez_id = xml_data[start_idx + 4:end_idx]
                    entrez_ids.append(entrez_id)
                    self._gene_cache[cache_key] = entrez_id
                else:
                    entrez_ids.append(None)
                    print(f"[HUMANBASE] WARNING: No Entrez ID found for {gene}", flush=True)
                    
            except Exception as e:
                print(f"[HUMANBASE] ERROR: Failed to get Entrez ID for {gene}: {e}", flush=True)
                entrez_ids.append(None)
                
            time.sleep(0.1)  # Rate limiting
        
        return [eid for eid in entrez_ids if eid is not None]
    
    def _make_request(self, url: str, params: Dict = None) -> Optional[Dict]:
        """Make HTTP request with retry logic"""
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    print(f"[HUMANBASE] WARNING: Request failed, retrying in {wait_time}s: {e}", flush=True)
                    time.sleep(wait_time)
                else:
                    print(f"[HUMANBASE] ERROR: Request failed after {self.max_retries} attempts: {e}", flush=True)
                    return None
    
    def _calculate_network_strength(self, interactions: List[Dict]) -> str:
        """Calculate overall network strength based on meaningful interactions"""
        if not interactions:
            return "minimal"
        
        # Only consider interactions with meaningful evidence
        meaningful_interactions = [
            edge for edge in interactions 
            if edge.get('weight', 0) >= 0.3 and self._has_meaningful_evidence(edge.get('evidence', {}))
        ]
        
        if not meaningful_interactions:
            return "minimal"
        
        weights = [edge.get('weight', 0) for edge in meaningful_interactions]
        avg_weight = sum(weights) / len(weights) if weights else 0
        
        if avg_weight >= 0.8:
            return "high"
        elif avg_weight >= 0.6:
            return "medium"
        elif avg_weight >= 0.4:
            return "low"
        else:
            return "minimal"
    
    def _has_meaningful_evidence(self, evidence: Dict) -> bool:
        """Check if interaction has meaningful evidence (not all zeros)"""
        if not evidence:
            return False
        
        # Check if any evidence type has a meaningful score
        meaningful_evidence = [
            score for score in evidence.values() 
            if isinstance(score, (int, float)) and score > 0.1
        ]
        
        return len(meaningful_evidence) > 0
    
    def _analyze_evidence_types(self, interactions: List[Dict]) -> str:
        """Analyze and summarize the types of evidence supporting interactions"""
        if not interactions:
            return ""
        
        evidence_counts = {}
        for interaction in interactions:
            evidence = interaction.get('evidence', {})
            for evidence_type, score in evidence.items():
                if isinstance(score, (int, float)) and score > 0.1:
                    evidence_counts[evidence_type] = evidence_counts.get(evidence_type, 0) + 1
        
        if not evidence_counts:
            return ""
        
        # Sort by frequency and take top 3
        top_evidence = sorted(evidence_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        evidence_names = [ev[0].lower().replace(' ', '_') for ev, count in top_evidence]
        
        if len(evidence_names) == 1:
            return evidence_names[0]
        elif len(evidence_names) == 2:
            return f"{evidence_names[0]} and {evidence_names[1]}"
        else:
            return f"{', '.join(evidence_names[:-1])}, and {evidence_names[-1]}"
    
    def _assess_tissue_specificity(self, tissue: str, interactions: List[Dict]) -> str:
        """Assess tissue specificity of interactions"""
        # Simplified heuristic based on interaction strength and type
        if not interactions:
            return "low"
        
        # Tissues with high specificity
        high_specificity_tissues = ['brain', 'heart', 'liver', 'kidney', 'muscle']
        medium_specificity_tissues = ['blood', 'lung', 'skin', 'bone']
        
        tissue_norm = self._normalize_tissue(tissue)
        
        if any(ht in tissue_norm for ht in high_specificity_tissues):
            return "high"
        elif any(mt in tissue_norm for mt in medium_specificity_tissues):
            return "medium"
        else:
            return "low"
    
    def analyze_tissue_network(self, genes: List[str], tissue: str, max_interactions: int = 10) -> HumanBaseResult:
        """Analyze tissue-specific gene interactions"""
        print(f"[HUMANBASE] INFO: Analyzing {self.get_interaction_type()} interactions for {len(genes)} genes in {tissue}", flush=True)
        
        # Validate tissue name first
        is_valid, tissue_norm, suggestions = self._validate_tissue(tissue)
        if not is_valid:
            error_msg = f"Invalid tissue '{tissue}' for HumanBase analysis."
            if suggestions:
                error_msg += f" Available options: {', '.join(suggestions)}"
            print(f"[HUMANBASE] ERROR: {error_msg}", flush=True)
            return HumanBaseResult(
                tissue=tissue,
                genes=genes,
                interaction_summary=f"Invalid tissue name: {tissue}. {error_msg}",
                network_strength="minimal",
                key_interactions=[],
                biological_processes=[],
                tissue_specificity="low",
                clinical_relevance=f"Unable to assess - {error_msg}"
            )
        
        # Get Entrez IDs
        entrez_ids = self._get_entrez_ids(genes)
        if not entrez_ids:
            print(f"[HUMANBASE] ERROR: No valid Entrez IDs found for genes: {genes}", flush=True)
            return HumanBaseResult(
                tissue=tissue,
                genes=genes,
                interaction_summary="No valid gene identifiers found",
                network_strength="minimal",
                key_interactions=[],
                biological_processes=[],
                tissue_specificity="low",
                clinical_relevance="Unable to assess - gene resolution failed"
            )
        
        interaction_type = self.get_interaction_type()
        
        # Get network data
        network_url = f"{self.base_url}/integrations/{tissue_norm}/network/"
        network_params = {
            'datatypes': interaction_type,
            'entrez': entrez_ids,  # Pass as list, requests will handle the formatting
            'node_size': max_interactions + 5  # Get a few extra for filtering
        }
        
        network_data = self._make_request(network_url, network_params)
        if not network_data:
            return HumanBaseResult(
                tissue=tissue,
                genes=genes,
                interaction_summary="Network data unavailable",
                network_strength="minimal",
                key_interactions=[],
                biological_processes=[],
                tissue_specificity="low",
                clinical_relevance="Unable to assess - network data unavailable"
            )
        
        # Process interactions with quality filtering
        interactions = []
        gene_map = {g['entrez']: g['standard_name'] for g in network_data.get('genes', [])}
        
        # First pass: collect all interactions and sort by weight
        all_edges = sorted(
            network_data.get('edges', []), 
            key=lambda x: x.get('weight', 0), 
            reverse=True
        )
        
        for edge in all_edges:
            # Skip very low weight interactions
            if edge.get('weight', 0) < 0.2:
                continue
                
            source_entrez = network_data['genes'][edge['source']]['entrez']
            target_entrez = network_data['genes'][edge['target']]['entrez']
            source_name = gene_map.get(source_entrez, f"Gene_{source_entrez}")
            target_name = gene_map.get(target_entrez, f"Gene_{target_entrez}")
            
            # Get detailed interaction evidence
            evidence_url = f"{self.base_url}/integrations/{tissue_norm}/evidence/"
            evidence_params = {
                'limit': 5,
                'source': source_entrez,
                'target': target_entrez
            }
            
            evidence_data = self._make_request(evidence_url, evidence_params)
            evidence_types = {}
            if evidence_data:
                evidence_types = {t['title']: round(t['weight'], 3) for t in evidence_data.get('datatypes', [])}
            
            # Only include interactions with meaningful evidence
            if self._has_meaningful_evidence(evidence_types):
                # Filter evidence to only include meaningful scores
                filtered_evidence = {
                    evidence_type: score 
                    for evidence_type, score in evidence_types.items()
                    if isinstance(score, (int, float)) and score > 0.1
                }
                
                interactions.append({
                    'source': source_name,
                    'target': target_name,
                    'weight': round(edge['weight'], 3),
                    'evidence': filtered_evidence
                })
                
                # Stop when we have enough high-quality interactions
                if len(interactions) >= max_interactions:
                    break
        
        # Get biological processes
        bp_url = f"{self.base_url}/terms/annotated/"
        bp_params = {
            'database': 'gene-ontology-bp',
            'entrez': entrez_ids,  # Pass as list, requests will handle the formatting
            'max_term_size': 15
        }
        
        bp_data = self._make_request(bp_url, bp_params)
        biological_processes = []
        if bp_data:
            biological_processes = [bp['title'] for bp in bp_data[:10]]  # Top 10 processes
        
        # Calculate metrics
        network_strength = self._calculate_network_strength(interactions)
        tissue_specificity = self._assess_tissue_specificity(tissue, interactions)
        
        # Generate summary and clinical relevance
        interaction_summary = self.interpret_interactions(interactions, tissue)
        clinical_relevance = self._assess_clinical_relevance(interactions, tissue, biological_processes)
        
        print(f"[HUMANBASE] INFO: Found {len(interactions)} {interaction_type} interactions with {network_strength} strength", flush=True)
        if biological_processes:
            print(f"[HUMANBASE] INFO: Identified {len(biological_processes)} relevant biological processes", flush=True)
        
        return HumanBaseResult(
            tissue=tissue,
            genes=genes,
            interaction_summary=interaction_summary,
            network_strength=network_strength,
            key_interactions=interactions,
            biological_processes=biological_processes,
            tissue_specificity=tissue_specificity,
            clinical_relevance=clinical_relevance
        )
    
    def _assess_clinical_relevance(self, interactions: List[Dict], tissue: str, processes: List[str]) -> str:
        """Assess clinical relevance of findings"""
        if not interactions:
            return "Limited clinical relevance - no significant interactions detected"
        
        # High-impact tissues for clinical relevance
        clinical_tissues = {
            'brain': 'neurological disorders',
            'heart': 'cardiovascular disease',
            'liver': 'metabolic disorders',
            'kidney': 'renal disease',
            'blood': 'hematological conditions',
            'lung': 'respiratory disorders',
            'muscle': 'muscular dystrophies'
        }
        
        tissue_norm = self._normalize_tissue(tissue)
        clinical_context = "general health"
        
        for clinical_tissue, context in clinical_tissues.items():
            if clinical_tissue in tissue_norm:
                clinical_context = context
                break
        
        strength = self._calculate_network_strength(interactions)
        
        if strength == "high":
            return f"High clinical relevance - strong interactions suggest therapeutic targets for {clinical_context}"
        elif strength == "medium":
            return f"Moderate clinical relevance - interactions may inform {clinical_context} mechanisms"
        elif strength == "low":
            return f"Low clinical relevance - weak interactions require validation for {clinical_context}"
        else:
            return f"Minimal clinical relevance - insufficient interaction strength for {clinical_context}"

class CoExpressionAnalyzer(BaseHumanBaseTool):
    """Analyzes tissue-specific gene co-expression patterns"""
    
    def get_interaction_type(self) -> str:
        return "co-expression"
    
    def interpret_interactions(self, interactions: List[Dict], tissue: str) -> str:
        if not interactions:
            return f"No significant co-expression patterns detected in {tissue}"
        
        strong_coexp = [i for i in interactions if i['weight'] >= 0.7]
        moderate_coexp = [i for i in interactions if 0.4 <= i['weight'] < 0.7]
        
        # Analyze evidence types
        evidence_summary = self._analyze_evidence_types(interactions)
        
        summary = f"Identified {len(interactions)} high-quality co-expression relationships in {tissue}. "
        
        if strong_coexp:
            summary += f"{len(strong_coexp)} show strong co-expression (≥0.7), suggesting coordinated regulation. "
        if moderate_coexp:
            summary += f"{len(moderate_coexp)} show moderate co-expression (0.4-0.7), indicating functional relationships. "
        
        if evidence_summary:
            summary += f"Primary evidence: {evidence_summary}."
        
        return summary

class ProteinInteractionAnalyzer(BaseHumanBaseTool):
    """Analyzes tissue-specific protein-protein interactions"""
    
    def get_interaction_type(self) -> str:
        return "interaction"
    
    def interpret_interactions(self, interactions: List[Dict], tissue: str) -> str:
        if not interactions:
            return f"No significant protein interactions detected in {tissue}"
        
        direct_interactions = [i for i in interactions if i['weight'] >= 0.6]
        evidence_summary = self._analyze_evidence_types(interactions)
        
        summary = f"Detected {len(interactions)} high-quality protein interactions in {tissue}. "
        
        if direct_interactions:
            summary += f"{len(direct_interactions)} represent high-confidence direct interactions (≥0.6), "
            summary += "indicating physical protein complexes or direct binding partners. "
        else:
            summary += "Interactions suggest pathway-level associations and functional relationships. "
        
        if evidence_summary:
            summary += f"Supported by {evidence_summary}."
        
        return summary

class TranscriptionFactorAnalyzer(BaseHumanBaseTool):
    """Analyzes tissue-specific transcription factor binding patterns"""
    
    def get_interaction_type(self) -> str:
        return "tf-binding"
    
    def interpret_interactions(self, interactions: List[Dict], tissue: str) -> str:
        if not interactions:
            return f"No significant transcription factor binding detected in {tissue}"
        
        tf_targets = {}
        for interaction in interactions:
            # Assume source is TF, target is gene (simplified)
            tf = interaction['source']
            if tf not in tf_targets:
                tf_targets[tf] = []
            tf_targets[tf].append(interaction['target'])
        
        evidence_summary = self._analyze_evidence_types(interactions)
        summary = f"Identified {len(interactions)} high-quality transcription factor-gene relationships in {tissue}. "
        
        if tf_targets:
            hub_tfs = [tf for tf, targets in tf_targets.items() if len(targets) >= 2]
            if hub_tfs:
                summary += f"Key regulatory hubs: {', '.join(hub_tfs[:3])} control multiple targets, "
                summary += "suggesting master regulatory roles in tissue-specific expression. "
        
        if evidence_summary:
            summary += f"Evidence includes {evidence_summary}."
        
        return summary

class MicroRNATargetAnalyzer(BaseHumanBaseTool):
    """Analyzes tissue-specific microRNA-target interactions"""
    
    def get_interaction_type(self) -> str:
        return "gsea-microrna-targets"
    
    def interpret_interactions(self, interactions: List[Dict], tissue: str) -> str:
        if not interactions:
            return f"No significant microRNA-target relationships detected in {tissue}"
        
        evidence_summary = self._analyze_evidence_types(interactions)
        summary = f"Found {len(interactions)} high-quality microRNA-target interactions in {tissue}. "
        summary += "These represent post-transcriptional regulatory mechanisms that fine-tune gene expression "
        summary += f"in {tissue}-specific contexts, controlling cellular responses and adaptation. "
        
        if evidence_summary:
            summary += f"Supported by {evidence_summary}."
        
        return summary

class PerturbationAnalyzer(BaseHumanBaseTool):
    """Analyzes tissue-specific gene perturbation outcomes"""
    
    def get_interaction_type(self) -> str:
        return "gsea-perturbations"
    
    def interpret_interactions(self, interactions: List[Dict], tissue: str) -> str:
        if not interactions:
            return f"No significant perturbation responses detected in {tissue}"
        
        high_impact = [i for i in interactions if i['weight'] >= 0.6]
        evidence_summary = self._analyze_evidence_types(interactions)
        
        summary = f"Identified {len(interactions)} high-quality perturbation-response relationships in {tissue}. "
        
        if high_impact:
            summary += f"{len(high_impact)} show high-impact responses (≥0.6), indicating genes with strong "
            summary += f"functional consequences when perturbed in {tissue}. These represent potential "
            summary += "therapeutic targets or biomarkers for tissue-specific interventions. "
        else:
            summary += f"Perturbations show moderate but significant effects in {tissue}, suggesting "
            summary += "important regulatory roles that merit further investigation. "
        
        if evidence_summary:
            summary += f"Evidence based on {evidence_summary}."
        
        return summary

# Convenience functions for agent-friendly access
def humanbase_analyze_tissue_coexpression(genes: List[str], tissue: str, max_interactions: int = 10) -> Tuple[str, str, List[Dict], List[str]]:
    """
    Analyze tissue-specific gene co-expression patterns
    Returns: (summary, confidence, top_interactions, biological_processes)
    """
    analyzer = CoExpressionAnalyzer()
    result = analyzer.analyze_tissue_network(genes, tissue, max_interactions)
    
    confidence = result.network_strength
    summary = result.interaction_summary
    top_interactions = result.key_interactions
    biological_processes = result.biological_processes
    
    return summary, confidence, top_interactions, biological_processes

def humanbase_analyze_tissue_protein_interactions(genes: List[str], tissue: str, max_interactions: int = 10) -> Tuple[str, str, List[Dict], List[str]]:
    """
    Analyze tissue-specific protein-protein interactions
    Returns: (summary, confidence, top_interactions, biological_processes)
    """
    analyzer = ProteinInteractionAnalyzer()
    result = analyzer.analyze_tissue_network(genes, tissue, max_interactions)
    
    confidence = result.network_strength
    summary = result.interaction_summary
    top_interactions = result.key_interactions
    biological_processes = result.biological_processes
    
    return summary, confidence, top_interactions, biological_processes

def humanbase_analyze_tissue_transcription_regulation(genes: List[str], tissue: str, max_interactions: int = 10) -> Tuple[str, str, List[Dict], List[str]]:
    """
    Analyze tissue-specific transcription factor regulation
    Returns: (summary, confidence, top_interactions, biological_processes)
    """
    analyzer = TranscriptionFactorAnalyzer()
    result = analyzer.analyze_tissue_network(genes, tissue, max_interactions)
    
    confidence = result.network_strength
    summary = result.interaction_summary
    top_interactions = result.key_interactions
    biological_processes = result.biological_processes
    
    return summary, confidence, top_interactions, biological_processes

def humanbase_analyze_tissue_microrna_regulation(genes: List[str], tissue: str, max_interactions: int = 10) -> Tuple[str, str, List[Dict], List[str]]:
    """
    Analyze tissue-specific microRNA-target regulation
    Returns: (summary, confidence, top_interactions, biological_processes)
    """
    analyzer = MicroRNATargetAnalyzer()
    result = analyzer.analyze_tissue_network(genes, tissue, max_interactions)
    
    confidence = result.network_strength
    summary = result.interaction_summary
    top_interactions = result.key_interactions
    biological_processes = result.biological_processes
    
    return summary, confidence, top_interactions, biological_processes

def humanbase_analyze_tissue_perturbation_outcomes(genes: List[str], tissue: str, max_interactions: int = 10) -> Tuple[str, str, List[Dict], List[str]]:
    """
    Analyze tissue-specific gene perturbation outcomes
    Returns: (summary, confidence, top_interactions, biological_processes)
    """
    analyzer = PerturbationAnalyzer()
    result = analyzer.analyze_tissue_network(genes, tissue, max_interactions)
    
    confidence = result.network_strength
    summary = result.interaction_summary
    top_interactions = result.key_interactions
    biological_processes = result.biological_processes
    
    return summary, confidence, top_interactions, biological_processes

def humanbase_analyze_comprehensive_tissue_network(genes: List[str], tissue: str, max_interactions: int = 8) -> Dict[str, Tuple[str, str, List[Dict], List[str]]]:
    """
    Comprehensive tissue-specific network analysis across all interaction types
    Returns: Dict with keys: coexpression, protein_interactions, tf_regulation, microrna_regulation, perturbation_outcomes
    """
    print(f"[HUMANBASE] INFO: Starting comprehensive tissue network analysis for {len(genes)} genes in {tissue}", flush=True)
    
    results = {}
    
    # Run all analyses
    analyzers = {
        'coexpression': CoExpressionAnalyzer(),
        'protein_interactions': ProteinInteractionAnalyzer(),
        'tf_regulation': TranscriptionFactorAnalyzer(),
        'microrna_regulation': MicroRNATargetAnalyzer(),
        'perturbation_outcomes': PerturbationAnalyzer()
    }
    
    for analysis_type, analyzer in analyzers.items():
        try:
            result = analyzer.analyze_tissue_network(genes, tissue, max_interactions)
            results[analysis_type] = (
                result.interaction_summary,
                result.network_strength,
                result.key_interactions,
                result.biological_processes
            )
        except Exception as e:
            print(f"[HUMANBASE] ERROR: {analysis_type} analysis failed: {e}", flush=True)
            results[analysis_type] = (
                f"Analysis failed: {str(e)}",
                "minimal",
                [],
                []
            )
    
    print(f"[HUMANBASE] INFO: Comprehensive analysis completed for {tissue}", flush=True)
    return results

# Legacy function for backward compatibility
def humanbase_ppi_retrieve(genes: list, tissue: str, max_node=10, interaction=None):
    """Legacy function - use class-based analyzers instead"""
    print("[HUMANBASE] WARNING: Using deprecated function. Consider using class-based analyzers.", flush=True)
    
    analyzer = ProteinInteractionAnalyzer()
    result = analyzer.analyze_tissue_network(genes, tissue, max_node)
    
    # Return simplified format for compatibility
    return result.key_interactions, result.biological_processes

def get_entrez_ids(gene_names):
    """Legacy function - use class methods instead"""
    analyzer = BaseHumanBaseTool()
    return analyzer._get_entrez_ids(gene_names)

#### Read Data Tools ####


def load_PPI_data(ppi_dir):
    ppi_layers = dict()
    for f in glob.glob(ppi_dir + "*"): # Expected format of filename: <PPI_DIR>/<CONTEXT>.<suffix>
        context = f.split(ppi_dir)[1].split(".")[0]
        # print(f"read: {f}")
        ppi = nx.read_edgelist(f)
        ppi_layers[context] = ppi
    return ppi_layers


def read_labels_from_evidence(positive_protein_prefix, negative_protein_prefix, raw_data_prefix, positive_proteins={}, negative_proteins={}, all_relevant_proteins={}):
    try:
        with open(positive_protein_prefix + '.json', 'r') as f:
            temp = json.load(f)
            positive_proteins = temp
        with open(negative_protein_prefix + '.json', 'r') as f:
            temp = json.load(f)
            negative_proteins = temp
        
        if raw_data_prefix != None:
            with open(raw_data_prefix + '.json', 'r') as f:
                temp = json.load(f)
                all_relevant_proteins = temp
        else: all_relevant_proteins = {}

        return positive_proteins, negative_proteins, all_relevant_proteins
    except:
        print("Files not found")
        return {}, {}, {}


def load_data(embed_path: str, labels_path: str, positive_proteins_prefix: str, negative_proteins_prefix: str, raw_data_prefix: str):
    
    embed = torch.load(embed_path)
    with open(labels_path, "r") as f:
        labels_dict = f.read()
    labels_dict = labels_dict.replace("\'", "\"")
    labels_dict = json.loads(labels_dict)
    celltypes = [c for c in labels_dict["Cell Type"] if c.startswith("CCI")]
    celltype_dict = {ct.split("CCI_")[1]: i for i, ct in enumerate(celltypes)}
    assert len(celltype_dict) > 0
    
    protein_names = []
    protein_celltypes = []
    for c, p in zip(labels_dict["Cell Type"], labels_dict["Name"]):
        if c.startswith("BTO") or c.startswith("CCI") or c.startswith("Sanity"): continue
        protein_names.append(p)
        protein_celltypes.append(c)

    proteins = pd.DataFrame.from_dict({"target": protein_names, "cell type": protein_celltypes})
    celltype_protein_dict = proteins.pivot_table(values="target", index="cell type", aggfunc={"target": list}).to_dict()["target"]
    assert len(celltype_protein_dict) > 0

    positive_proteins, negative_proteins, all_relevant_proteins = read_labels_from_evidence(positive_proteins_prefix, negative_proteins_prefix, raw_data_prefix)
    assert len(positive_proteins) > 0

    return embed, celltype_dict, celltype_protein_dict, positive_proteins, negative_proteins, all_relevant_proteins

