import requests
import urllib.parse
from typing import Dict, List, Optional, Union, ClassVar, Any, Iterator, Iterable, Tuple
import sys, os

from tqdm import tqdm
from nltk import sent_tokenize
from .gpt_utils import chat_completion
import time
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_community.utilities.pubmed import PubMedAPIWrapper
from langchain_core.documents import Document
import urllib
import json

import warnings

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup

# FlagEmbedding may download models on first import - this can take time
print(
    "[INFO] Loading FlagEmbedding library (may download reranker models)...", flush=True
)
from FlagEmbedding import FlagReranker

# print("[INFO] FlagEmbedding loaded successfully", flush=True)

from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from .gpt_utils import chat_completion

# from .open_scholar import OpenScholar, search_paper_via_query
from . import instructions


class OpenAlexTool:
    """
    Tool to retrieve literature from OpenAlex based on search keywords.

    This class provides methods to search for academic papers using the OpenAlex API
    with title and abstract search functionality.
    """

    BASE_URL: ClassVar[str] = "https://api.openalex.org/works"
    DEFAULT_EMAIL: ClassVar[str] = "research@example.com"
    MAX_PER_PAGE: ClassVar[int] = 200
    DEFAULT_MAX_RESULTS: ClassVar[int] = 5

    def __init__(self, email: str = DEFAULT_EMAIL):
        """
        Initialize the OpenAlex tool.

        Args:
            email: Email address for polite pool access (optional)
        """
        self.base_url = self.BASE_URL
        self.email = email

    def run(self, arguments: Dict[str, Any]) -> Union[List[Dict[str, Any]], str]:
        """
        Main entry point for the tool.

        Args:
            arguments: Dictionary containing search parameters

        Returns:
            List of paper dictionaries or error message
        """
        search_keywords = arguments.get("search_keywords")
        if not search_keywords:
            return "Error: search_keywords parameter is required"

        max_results = arguments.get("max_results", self.DEFAULT_MAX_RESULTS)
        year_from = arguments.get("year_from")
        year_to = arguments.get("year_to")
        open_access = arguments.get("open_access")

        return self.search_literature(
            search_keywords, max_results, year_from, year_to, open_access
        )

    def search_literature(
        self,
        search_keywords: str,
        max_results: int = DEFAULT_MAX_RESULTS,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        open_access: Optional[bool] = None,
        verbose: bool = True,
    ) -> Union[List[Dict[str, Any]], str]:
        """
        Search for literature using OpenAlex API with title and abstract search.

        Args:
            search_keywords: Keywords to search for in title and abstract
            max_results: Maximum number of results to return (default: 10)
            year_from: Start year for publication date filter (optional)
            year_to: End year for publication date filter (optional)
            open_access: Filter for open access papers only (optional)

        Returns:
            List of dictionaries containing paper information or error message
        """
        if not search_keywords or not search_keywords.strip():
            return "Error: search_keywords cannot be empty"

        try:
            params = self._build_search_params(
                search_keywords, max_results, year_from, year_to, open_access
            )
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()

            # Parse response with error handling
            data = response.json()
            if not data or not isinstance(data, dict):
                return "Error: Invalid response format from OpenAlex API"

            results = data.get("results", [])
            if not results:
                return f"No results found for search keywords: '{search_keywords}'"

            # Extract paper info with error handling
            papers = []
            for work in results:
                if work and isinstance(
                    work, dict
                ):  # Ensure work is not None and is a dict
                    try:
                        paper_info = self._extract_paper_info(work)
                        if paper_info:  # Only add if extraction was successful
                            papers.append(paper_info)
                    except Exception as e:
                        print(f"[OpenAlex] Warning: Failed to extract paper info: {e}")
                        continue

            if not papers:
                return f"Error: No valid papers could be extracted from search results"

            if verbose:
                print(
                    f"[OpenAlex] Retrieved {len(papers)} papers for keywords: '{search_keywords}'"
                )
            return papers

        except requests.exceptions.Timeout:
            return "Error: Request timeout - OpenAlex API did not respond in time"
        except requests.exceptions.RequestException as e:
            return f"Error retrieving data from OpenAlex: {e}"
        except Exception as e:
            return f"Unexpected error during search: {e}"

    def _build_search_params(
        self,
        search_keywords: str,
        max_results: int,
        year_from: Optional[int],
        year_to: Optional[int],
        open_access: Optional[bool],
    ) -> Dict[str, str]:
        """Build query parameters for OpenAlex API request with title and abstract search."""
        # Clean and prepare keywords
        keywords = search_keywords.strip()

        # Build search query - use the filter approach for title_and_abstract.search
        params = {
            "filter": f"title_and_abstract.search:{keywords}",
            "per-page": str(min(max_results, self.MAX_PER_PAGE)),
            "sort": "relevance_score:desc",  # Sort by OpenAlex relevance score
            "mailto": self.email,  # Required for OpenAlex API access
        }

        # Build additional filters
        filters = [f"title_and_abstract.search:{keywords}"]

        # Add year filters
        if year_from is not None and year_to is not None:
            filters.append(f"publication_year:{year_from}-{year_to}")
        elif year_from is not None:
            filters.append(f"publication_year:>{year_from-1}")
        elif year_to is not None:
            filters.append(f"publication_year:<{year_to+1}")

        # Add open access filter
        if open_access is True:
            filters.append("is_oa:true")
        elif open_access is False:
            filters.append("is_oa:false")

        # Combine all filters
        params["filter"] = ",".join(filters)

        return params

    def _extract_paper_info(self, work: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract relevant information from a work object returned by OpenAlex API.

        Args:
            work: Work object from OpenAlex API response

        Returns:
            Formatted paper information dictionary matching semantic scholar format or None if authors are empty
        """
        if not work or not isinstance(work, dict):
            return None

        try:
            # Extract basic information with safe defaults
            title = work.get("title", "No title available") or "No title available"
            publication_year = (
                work.get("publication_year", "Year not available")
                or "Year not available"
            )

            # Extract and reconstruct abstract
            abstract = self._reconstruct_abstract(work.get("abstract_inverted_index"))

            # Extract authors in semantic scholar format
            authors = self._extract_authors_semantic_scholar_format(
                work.get("authorships", []) or []
            )

            # Filter: Skip papers with no authors
            if not authors or len(authors) == 0:
                return None

            # Extract publication details
            primary_location = work.get("primary_location") or {}
            venue = "Unknown venue"
            if primary_location and isinstance(primary_location, dict):
                source = primary_location.get("source") or {}
                if isinstance(source, dict):
                    venue = (
                        source.get("display_name", "Unknown venue") or "Unknown venue"
                    )

            doi = work.get("doi", "No DOI") or "No DOI"
            citation_count = work.get("cited_by_count", 0) or 0

            # Extract open access information
            open_access_info = work.get("open_access") or {}
            is_open_access = False
            pdf_url = None
            if isinstance(open_access_info, dict):
                is_open_access = open_access_info.get("is_oa", False) or False
                pdf_url = open_access_info.get("oa_url")

            # Build URL - prefer DOI, fallback to OpenAlex ID
            url = doi if doi != "No DOI" else work.get("id", "") or ""
            openalex_id = work.get("id", "") or ""

            # Get OpenAlex relevance score if available
            relevance_score = work.get("relevance_score", 0.0) or 0.0

            # Return in semantic scholar compatible format
            return {
                "semantic_scholar_id": openalex_id,  # Use OpenAlex ID as identifier
                "type": "openalex_abstract",
                "year": publication_year,
                "authors": authors,
                "title": title,
                "text": abstract,  # Abstract as text field
                "url": url,
                "citation_counts": citation_count,  # Match semantic scholar field name
                "abstract": abstract,  # Also keep separate abstract field
                # Additional OpenAlex-specific fields
                "venue": venue,
                "doi": doi,
                "open_access": is_open_access,
                "pdf_url": pdf_url,
                "openalex_id": openalex_id,
                "relevance_score": relevance_score,
            }
        except Exception as e:
            print(f"[OpenAlex] Error extracting paper info: {e}")
            return None

    def _reconstruct_abstract(
        self, abstract_inverted_index: Optional[Dict[str, List[int]]]
    ) -> str:
        """
        Reconstruct abstract from inverted index.

        Args:
            abstract_inverted_index: Dictionary mapping words to their positions

        Returns:
            Reconstructed abstract text
        """
        if not abstract_inverted_index:
            return "Abstract not available"

        try:
            # Find the maximum position to determine array size
            max_position = 0
            for positions in abstract_inverted_index.values():
                if positions:
                    max_position = max(max_position, max(positions))

            # Create array with appropriate size
            abstract_words = [""] * (max_position + 1)

            # Fill in the words at their positions
            for word, positions in abstract_inverted_index.items():
                for pos in positions:
                    if 0 <= pos < len(abstract_words):
                        abstract_words[pos] = word

            # Join words and clean up
            abstract = " ".join(word for word in abstract_words if word).strip()
            return abstract if abstract else "Abstract not available"

        except Exception:
            return "Abstract reconstruction failed"

    def _extract_authors_semantic_scholar_format(
        self, authorships: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Extract author names from authorships in semantic scholar format."""
        authors = []
        for authorship in authorships:
            author = authorship.get("author", {})
            author_name = author.get("display_name", "Unknown Author")
            # Match semantic scholar format with authorId and name
            authors.append(
                {
                    "authorId": author.get("id", "") or "",  # OpenAlex author ID
                    "name": author_name,
                }
            )
        return authors

    def _extract_authors(self, authorships: List[Dict[str, Any]]) -> List[str]:
        """Extract author names from authorships (legacy method)."""
        authors = []
        for authorship in authorships:
            author = authorship.get("author", {})
            author_name = author.get("display_name", "Unknown Author")
            authors.append(author_name)
        return authors

    def _extract_organizations(self, authorships: List[Dict[str, Any]]) -> List[str]:
        """Extract organization names from authorships."""
        organizations = set()
        for authorship in authorships:
            for institution in authorship.get("institutions", []):
                org_name = institution.get("display_name")
                if org_name:
                    organizations.add(org_name)
        return list(organizations)


def paper_search_from_openalex(
    search_keywords: str,
    max_results: int = 10,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    open_access: Optional[bool] = None,
) -> Union[List[Dict[str, Any]], str]:
    """
    Search for literature using OpenAlex API with title and abstract search.

    Args:
        search_keywords: Keywords to search for
        max_results: Maximum number of results to return
        year_from: Start year for publication date filter
        year_to: End year for publication date filter
        open_access: Filter for open access papers only

    Returns:
        List of paper dictionaries in semantic scholar format or error message
    """
    openalex = OpenAlexTool()
    return openalex.search_literature(
        search_keywords, max_results, year_from, year_to, open_access
    )


def search_openalex_papers(
    question: str,
    max_results: int = 5,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    open_access: Optional[bool] = None,
    model_name: str = "gpt-4o",
    verbose: bool = True,
) -> tuple[List[Dict[str, Any]], List[str]]:
    """
    Search for literature using OpenAlex API with intelligent keyword generation.

    Args:
        question: Search question/keywords
        max_results: Maximum number of results to return
        year_from: Start year for publication date filter
        year_to: End year for publication date filter
        open_access: Filter for open access papers only
        model_name: Model for keyword generation

    Returns:
        Tuple of (paper_list, keywords_used) matching semantic scholar format
    """
    # Import here to avoid circular imports
    try:
        from tool_space.search_api import retrieve_keywords

        # Generate optimized keywords for OpenAlex
        if verbose:
            print(
                f"[SEARCH_OPENALEX] INFO: Generating keywords for question: '{question}'"
            )
        optimized_keywords = retrieve_keywords(
            question,
            model_name=model_name,
            platform="openalex",
            query_num=4,
            verbose=verbose,
        )
        if verbose:
            print(f"[SEARCH_OPENALEX] INFO: Using keywords: {optimized_keywords}")

    except Exception as e:
        if verbose:
            print(
                f"[SEARCH_OPENALEX] WARNING: Keyword generation failed ({e}), using question directly"
            )
        optimized_keywords = [question.strip()]

    openalex = OpenAlexTool()
    all_papers = {}
    successful_queries = 0

    # Search with each keyword
    for i, keyword in enumerate(optimized_keywords):
        if verbose:
            print(
                f"[SEARCH_OPENALEX] DEBUG: Processing keyword {i+1}/{len(optimized_keywords)}: '{keyword}'"
            )

        results = openalex.search_literature(
            keyword, max_results, year_from, year_to, open_access, verbose
        )

        # If error occurred, try next keyword
        if isinstance(results, str):
            if verbose:
                print(
                    f"[SEARCH_OPENALEX] WARNING: No papers found for keyword: '{keyword}' - {results}"
                )
            continue

        successful_queries += 1
        if verbose:
            print(
                f"[SEARCH_OPENALEX] SUCCESS: Found {len(results)} papers for keyword: '{keyword}'"
            )

        # Deduplicate by OpenAlex ID
        for paper in results:
            paper_id = paper.get("openalex_id", paper.get("semantic_scholar_id", ""))
            if paper_id and paper_id not in all_papers:
                all_papers[paper_id] = paper

    if successful_queries == 0:
        if verbose:
            print(f"[SEARCH_OPENALEX] ERROR: No successful queries found")
        return [], []

    # Convert to list and sort by relevance score if available
    final_papers = list(all_papers.values())
    try:
        final_papers.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    except:
        pass  # Sort failed, continue with unsorted list

    # Limit to max_results
    final_papers = final_papers[:max_results]

    if verbose:
        print(
            f"[SEARCH_OPENALEX] SUMMARY: {successful_queries}/{len(optimized_keywords)} queries successful, {len(final_papers)} unique papers found"
        )

    return final_papers, optimized_keywords


#### Open Scholar ####

# nlp = spacy.load('en_core_web_sm')


def remove_citations(sent):
    return (
        re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent))
        .replace(" |", "")
        .replace("]", "")
    )


def rerank_paragraphs_bge(
    query,
    paragraphs,
    reranker,
    norm_cite=False,
    start_index=0,
    use_abstract=False,
    batch_size=50,
):
    """
    Re-rank paragraphs using a BGE-based reranker, optionally adjusting scores based on citation counts.

    Parameters:
        query (str): The query text.
        paragraphs (list of dict): List of paragraph dictionaries; each must contain a "text" field.
        reranker (object): An object with a `compute_score` method that accepts a list of [query, text] pairs.
        norm_cite (bool): If True, adjust scores by adding a normalized citation count.
        start_index (int): Starting index for ranking (currently unused, reserved for future extensions).
        use_abstract (bool): If True, include the abstract (and title) in the text used for scoring.

    Returns:
        new_orders (list): Reordered list of paragraph dictionaries, ranked from highest to lowest score.
        result_dic (dict): Mapping from original paragraph index to computed score.
        id_mapping (dict): Mapping from new order index to original paragraph index.
    """
    # Filter out paragraphs with missing text
    valid_paragraphs = [p for p in paragraphs if p.get("text") is not None]

    # Prepare text input for each paragraph based on the use_abstract flag
    if use_abstract:
        paragraph_texts = [
            (
                f"{p.get('title', '')}\n{p.get('abstract', '')}\n{p['text']}"
                if p.get("title") is not None and p.get("abstract") is not None
                else p["text"]
            )
            for p in valid_paragraphs
        ]
    else:
        paragraph_texts = [
            (
                f"{p.get('title', '')} {p['text']}"
                if p.get("title") is not None
                else p["text"]
            )
            for p in valid_paragraphs
        ]

    # Compute scores using the reranker; each input is a pair [query, paragraph_text]
    scores = reranker.compute_score(
        [[query, text] for text in paragraph_texts], batch_size=batch_size
    )

    # Wrap a single float score in a dictionary; otherwise, enumerate the scores
    if isinstance(scores, float):
        result_dic = {0: scores}
    else:
        result_dic = {idx: score for idx, score in enumerate(scores)}

    # Optionally adjust scores based on normalized citation counts, if available
    if norm_cite:
        # Extract citation counts for paragraphs that include them
        citation_counts = [
            p["citation_counts"]
            for p in valid_paragraphs
            if p.get("citation_counts") is not None
        ]
        if citation_counts:
            max_citations = max(citation_counts)
            # Update scores by adding the normalized citation count to each paragraph's score
            for idx, p in enumerate(valid_paragraphs):
                if p.get("citation_counts") is not None:
                    result_dic[idx] += p["citation_counts"] / max_citations

    # Sort paragraphs by score in descending order
    sorted_scores = sorted(result_dic.items(), key=lambda x: x[1], reverse=True)

    # Reorder paragraphs and build an index mapping (new index -> original index)
    new_orders = []
    id_mapping = {}
    for new_idx, (orig_idx, _) in enumerate(sorted_scores):
        new_orders.append(valid_paragraphs[orig_idx])
        id_mapping[new_idx] = orig_idx

    return new_orders, result_dic, id_mapping


def search_paper_via_query(query, max_paper_num=2, attempt=3, minCitationCount=5):
    if "Search queries: " in query:
        query = query.split("Search queries: ")[1]

    paper_field_collection = "title,year,abstract,authors.name,citationCount,year,url,externalIds,publicationVenue"
    query_params = {
        "query": query,
        "limit": max_paper_num,
        "minCitationCount": minCitationCount,
        "sort": "citationCount:desc",
        "fields": paper_field_collection,
    }
    # api_key = "19dnRruThD7a8AusEjLna3dgPtPx2vlH8bCsfite"
    # Send the API request

    while attempt > 0:
        response = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search", params=query_params
        )
        time.sleep(2)

        if response.status_code == 200:
            response_data = response.json()
            break
        # Process and print the response data as needed
        else:
            attempt -= 1
            response_data = None
            print(
                f"Request failed with status code {response.status_code}: {response.text}"
            )
            time.sleep(5)
    # except:
    # response_data = None
    if response_data is None or len(response_data) == 0 or "data" not in response_data:
        # print(f"retrieval failed: {response_data}")
        return None
    else:
        return response_data["data"]


def create_prompt_with_llama3_format(
    prompt,
    system_message="You are a helpful AI assistant for scientific literature review. Please carefully follow user's instruction and help them to understand the most recent papers.",
):
    if system_message is not None:
        formatted_text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{0}<|eot_id|>".format(
            system_message
        )
    else:
        formatted_text = "<|begin_of_text|>"
    formatted_text += (
        "<|start_header_id|>user<|end_header_id|>\n\n" + prompt + "<|eot_id|>"
    )
    formatted_text += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return formatted_text


class OpenScholar(object):
    def __init__(
        self,
        model,
        tokenizer,
        client_llm=None,
        use_contexts=True,
        top_n=8,
        reranker=None,
        min_citation=None,
        norm_cite=False,
        ss_retriever=False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.client_llm = client_llm
        self.top_n = top_n
        self.no_retrieval = not use_contexts
        self.reranker = reranker
        self.min_citation = min_citation
        self.norm_cite = norm_cite
        self.ss_retriever = ss_retriever
        self.use_contexts = use_contexts

    # Reranking: We rerank passages based on the LMs' predictions on how useful passages are.
    def process_ranking_results(self, result):
        ratings = {
            int(match.group(1)): int(match.group(2))
            for match in re.finditer(r"\[(\d+)\] Rating: (\d)", result)
        }
        return ratings

    def reranking_passages_cross_encoder(self, item, use_abstract=False):

        if self.min_citation is not None:
            ctx_above_threshold = [
                p
                for p in item["ctxs"]
                if "citation_counts" in p and p["citation_counts"] >= self.min_citation
            ]
            if len(ctx_above_threshold) > self.top_n:
                item["ctxs"] = ctx_above_threshold
                # print("after filtering -- number of ctxs: {0}".format(len(item["ctxs"])))

        reranked_contexts, sorted_results, id_mapping = rerank_paragraphs_bge(
            item["input"],
            item["ctxs"],
            self.reranker,
            norm_cite=self.norm_cite,
            use_abstract=use_abstract,
        )
        return reranked_contexts, sorted_results, id_mapping

    def reranking_passages_cross_encoder_supplemental(self, item, passages):

        if self.min_citation is not None:
            ctx_above_threshold = [
                p
                for p in passages
                if "citation_counts" in p and p["citation_counts"] >= self.min_citation
            ]
            if len(ctx_above_threshold) > self.top_n:
                passages = ctx_above_threshold
                # print("after filtering -- number of ctxs: {0}".format(len(passages)))

        reranked_contexts, sorted_results, id_mapping = rerank_paragraphs_bge(
            item["input"],
            passages,
            self.reranker,
            norm_cite=False,
            start_index=len(item["ctxs"]),
        )

        return reranked_contexts, sorted_results, id_mapping

    def retrieve_keywords(self, question):
        prompt = [
            instructions.keyword_extraction_prompt.format_map({"question": question})
        ]

        # Use default client
        raw_output = chat_completion(prompt[0], model=self.client_llm)
        outputs = raw_output
        raw_output = (
            [
                t.split("[Response_End]")[0]
                for t in outputs.split("[Response_Start]")
                if "[Response_End]" in t
            ][0]
            if "[Response_End]" in outputs
            else outputs
        )

        queries = raw_output.split(", ")[:3]
        queries = [
            query.replace("Search queries: ", "") for query in queries if len(query) > 0
        ]
        return queries

    # Generation: Generate output based on query, passages
    def generate_response(
        self,
        item,
        max_tokens=3000,
        llama3_chat=False,
        task_name="default",
        zero_shot=False,
    ):
        ranked_results = {}
        # print("zero-shot?: {}".format(zero_shot))
        if self.use_contexts is False:
            ctxs = []
            # support more task
            if task_name in instructions.task_instructions:
                if zero_shot is True:
                    input_query = (
                        instructions.task_instructions[task_name][0]
                        + instructions.task_instructions[task_name][1]
                        + item["input"]
                    )
                else:
                    demonstration = instructions.demonstrations[task_name]
                    input_query = (
                        instructions.task_instructions[task_name][0]
                        + demonstration
                        + instructions.task_instructions[task_name][1]
                        + item["input"]
                    )
            if task_name == "single_qa":
                input_query = instructions.generation_instance_prompts_w_references_single_paper_no_context.format_map(
                    {"input": item["input"]}
                )
            else:
                if zero_shot is True:
                    input_query = instructions.generation_instance_prompts_w_references_zero_shot.format_map(
                        {"context": ctxs, "input": item["input"]}
                    )
                else:
                    input_query = instructions.generation_instance_prompts_w_references.format_map(
                        {"context": ctxs, "input": item["input"]}
                    )
            item["final_passages"] = ctxs
        else:
            ctxs = ""
            for doc_idx, doc in enumerate(item["ctxs"][: self.top_n]):
                if "title" in doc and len(doc["title"]) > 0:
                    ctxs += "[{0}] Title: {1} Text: {2}\n".format(
                        doc_idx, doc["title"], doc["text"]
                    )
                else:
                    ctxs += "[{0}] {1}\n".format(doc_idx, doc["text"])
            item["final_passages"] = ctxs

            if task_name == "summarization":
                if zero_shot is True:
                    input_query = instructions.prompts_w_references_summarization_zero_shot.format_map(
                        {"context": ctxs, "input": item["input"]}
                    )
                else:
                    input_query = instructions.generation_instance_prompts_summarization.format_map(
                        {"context": ctxs, "input": item["input"]}
                    )
            elif task_name == "single_qa":
                if zero_shot is True:
                    input_query = instructions.generation_instance_prompts_w_references_single_paper_zero_shot.format_map(
                        {"context": ctxs, "input": item["input"]}
                    )
                else:
                    input_query = instructions.generation_instance_prompts_w_references_single_paper.format_map(
                        {"context": ctxs, "input": item["input"]}
                    )

            elif task_name in instructions.task_instructions:
                task_instruction = instructions.task_instructions[task_name][0]
                instance_header = instructions.task_instructions[task_name][1]
                if zero_shot is True:
                    input_query = "{0}\nReferences:\n{1}\n{2}{3}".format(
                        task_instruction, ctxs, instance_header, item["input"]
                    )
                else:
                    demonstration = instructions.demonstrations[task_name]
                    input_query = "{0}{1}\nReferences:\n{2}\n{3}{4}".format(
                        task_instruction,
                        demonstration,
                        ctxs,
                        instance_header,
                        item["input"],
                    )

            else:
                if zero_shot is True:
                    input_query = instructions.generation_instance_prompts_w_references_zero_shot.format_map(
                        {"context": ctxs, "input": item["input"]}
                    )
                else:
                    input_query = instructions.generation_instance_prompts_w_references.format_map(
                        {"context": ctxs, "input": item["input"]}
                    )

        if llama3_chat is True:
            input_query = create_prompt_with_llama3_format(input_query)

        outputs = chat_completion(input_query, model=self.client_llm)
        raw_output = (
            [
                t.split("[Response_End]")[0]
                for t in outputs.split("[Response_Start]")
                if "[Response_End]" in t
            ][0]
            if "[Response_End]" in outputs
            else outputs
        )

        if "References:" in raw_output:
            raw_output = raw_output.split("References:")[0]
        item["output"] = raw_output
        return raw_output, ctxs

    # Feedback: send feedback on model' predictions.
    def process_feedback(self, response):
        feedbacks_and_questions = re.findall(
            r"Feedback: (.*?)(?:Question: (.*?))?\n", response
        )
        ratings = [
            (feedback.strip(), question.strip() if question else "")
            for feedback, question in feedbacks_and_questions
        ]
        return ratings

    def get_feedback(self, item, llama3_chat):
        input_query = instructions.feedback_example_instance_prompt.format_map(
            {
                "question": item["input"],
                "passages": item["final_passages"],
                "answer": item["output"],
            }
        )
        # TODO: check if the llama3 chat format is helpful or not.
        if llama3_chat is True:
            input_query = create_prompt_with_llama3_format(input_query)

        outputs = chat_completion(input_query, model=self.client_llm)
        raw_output = (
            [
                t.split("[Response_End]")[0]
                for t in outputs.split("[Response_Start]")
                if "[Response_End]" in t
            ][0]
            if "[Response_End]" in outputs
            else outputs
        )
        feedbacks = self.process_feedback(raw_output)
        return feedbacks

    def edit_with_feedback(self, item, feedback, max_tokens=3000, llama3_chat=False):
        input_query = instructions.editing_instance_prompt.format_map(
            {
                "question": item["input"],
                "passages": item["final_passages"],
                "answer": item["output"],
                "feedback": feedback,
            }
        )

        # TODO: check if the llama3 chat format is helpful or not.
        if llama3_chat is True:
            input_query = create_prompt_with_llama3_format(input_query)

        outputs = chat_completion(input_query, model=self.client_llm)
        raw_output = (
            [
                t.split("[Response_End]")[0]
                for t in outputs.split("[Response_Start]")
                if "[Response_End]" in t
            ][0]
            if "[Response_End]" in outputs
            else outputs
        )
        return raw_output

    def edit_with_feedback_retrieval(
        self,
        item,
        feedback,
        passages,
        passage_start_index,
        max_tokens=2000,
        llama3_chat=False,
    ):
        processed_passages = ""
        for doc_idx, doc in enumerate(passages[: self.top_n]):
            if "title" in doc and len(doc["title"]) > 0:
                processed_passages += "[{0}] Title: {1} Text: {2}\n".format(
                    passage_start_index + doc_idx, doc["title"], doc["text"]
                )
            else:
                processed_passages += "[{0}] {1}\n".format(
                    passage_start_index + doc_idx + len(item["ctxs"]), doc["text"]
                )

        input_query = instructions.editing_with_retrieval_instance_prompt.format_map(
            {
                "question": item["input"],
                "retrieved_passages": processed_passages,
                "answer": item["output"],
                "feedback": feedback,
            }
        )
        if llama3_chat is True:
            input_query = create_prompt_with_llama3_format(input_query)

        outputs = chat_completion(input_query, model=self.client_llm)
        raw_output = (
            [
                t.split("[Response_End]")[0]
                for t in outputs.split("[Response_Start]")
                if "[Response_End]" in t
            ][0]
            if "[Response_End]" in outputs
            else outputs
        )
        return raw_output

    def insert_attributions_posthoc_paragraph(self, item, llama3_chat=False):
        text = item["output"]
        if "final_passages" in item:
            passages = item["final_passages"]
        else:
            ctxs = item["ctxs"]
            passages = ""
            for idx, p in enumerate(ctxs):
                passages += "[{0}] {1}\n".format(idx, p)

        # print(text)
        sentences = text.split("\n")
        # print(sentences)
        # post process sentences
        updated_sentences = []
        post_hoc_sentence = {}

        for s_index, statement in enumerate(sentences):
            if len(statement) < 10:
                if (
                    len(updated_sentences) > 0
                    and len(statement) > 0
                    and statement[0] == "["
                ):
                    updated_sentences[-1] = updated_sentences[-1] + " " + statement
                else:
                    updated_sentences.append(statement)

            else:
                # cases where citations are included
                if "[" in statement or (
                    s_index < len(sentences) - 1
                    and len(sentences[s_index + 1]) > 0
                    and sentences[s_index + 1][0] == "["
                ):
                    updated_sentences.append(statement)
                else:
                    updated_sentences.append("[replace_{}]".format(s_index))
                    post_hoc_sentence["[replace_{}]".format(s_index)] = statement

        if len(post_hoc_sentence) > 0:
            print(
                "{0} sentences require attributions, e..g, {1}".format(
                    len(post_hoc_sentence), list(post_hoc_sentence.values())[0]
                )
            )
            prompts = []
            for s in list(post_hoc_sentence.values()):
                input_query = instructions.posthoc_attributions_paragraph.format_map(
                    {"statement": s, "passages": passages}
                )
                if llama3_chat is True:
                    input_query = create_prompt_with_llama3_format(input_query)

                prompts.append(input_query)

            outputs = []
            for input_query in prompts:
                raw_output = chat_completion(input_query, model=self.client_llm)
                outputs.append(raw_output)

            # Postprocess Output
            for output, sentence_key in zip(outputs, list(post_hoc_sentence.keys())):
                if (
                    len(
                        [
                            t.split("[Response_End]")[0]
                            for t in output.split("[Response_Start]")
                            if "[Response_End]" in t
                        ]
                    )
                    == 0
                ):
                    post_hoc_sentence[sentence_key] = post_hoc_sentence[sentence_key]
                else:
                    processed_output = [
                        t.split("[Response_End]")[0]
                        for t in output.split("[Response_Start]")
                        if "[Response_End]" in t
                    ][0]
                    post_hoc_sentence[sentence_key] = processed_output

            final_processed_outputs = []
            for item in updated_sentences:
                if item in post_hoc_sentence:
                    final_processed_outputs.append(post_hoc_sentence[item])
                else:
                    final_processed_outputs.append(item)
            updated_sentences = final_processed_outputs

        return "\n".join(updated_sentences)

    def insert_attributions_posthoc(self, item, llama3_chat=False):
        text = item["output"]
        passages = item["final_passages"]

        sentences = sent_tokenize(text)
        # post process sentences
        updated_sentences = []
        post_hoc_sentence = {}

        for s_index, statement in enumerate(sentences):
            if len(statement) < 10:
                if statement[0] == "[":
                    updated_sentences[-1] = updated_sentences[-1] + " " + statement
                else:
                    updated_sentences.append(statement)

            else:
                # cases where citations are included
                if "[" in statement or (
                    s_index < len(sentences) - 1 and sentences[s_index + 1][0] == "["
                ):
                    updated_sentences.append(statement)
                else:
                    updated_sentences.append("[replace_{}]".format(s_index))
                    post_hoc_sentence["[replace_{}]".format(s_index)] = statement

        if len(post_hoc_sentence) > 0:

            print(
                "{0} sentences require attributions, e..g, {1}".format(
                    len(post_hoc_sentence), list(post_hoc_sentence.values())[0]
                )
            )
            prompts = []
            for s in list(post_hoc_sentence.values()):
                input_query = instructions.posthoc_attributions.format_map(
                    {"statement": s, "passages": passages}
                )

                if llama3_chat is True:
                    input_query = create_prompt_with_llama3_format(input_query)

                prompts.append(input_query)

            outputs = []
            for input_query in prompts:
                raw_output = chat_completion(input_query, model=self.client_llm)
                outputs.append(raw_output)

            # process_output
            for output, sentence_key in zip(outputs, list(post_hoc_sentence.keys())):
                if (
                    len(
                        [
                            t.split("[Response_End]")[0]
                            for t in output.split("[Response_Start]")
                            if "[Response_End]" in t
                        ]
                    )
                    == 0
                ):
                    post_hoc_sentence[sentence_key] = post_hoc_sentence[sentence_key]
                else:
                    processed_output = [
                        t.split("[Response_End]")[0]
                        for t in output.split("[Response_Start]")
                        if "[Response_End]" in t
                    ][0]
                    post_hoc_sentence[sentence_key] = processed_output

            final_processed_outputs = []
            for item in updated_sentences:
                if item in post_hoc_sentence:
                    final_processed_outputs.append(post_hoc_sentence[item])
                else:
                    final_processed_outputs.append(item)
            updated_sentences = final_processed_outputs

        return " ".join(updated_sentences)

    def insert_attributions_posthoc_paragraph_all(self, item, llama3_chat=False):
        text = item["output"]
        if "final_passages" in item:
            passages = item["final_passages"]
        else:
            ctxs = item["ctxs"]
            passages = ""
            for idx, p in enumerate(ctxs):
                passages += "[{0}] {1}\n".format(idx, p)

        sentences = text.split("\n")
        # print(sentences)
        updated_sentences = []
        post_hoc_sentence = {}
        prompts = []

        for s_index, statement in enumerate(sentences):
            if len(statement) < 10:
                if (
                    len(updated_sentences) > 0
                    and len(statement) > 0
                    and statement[0] == "["
                ):
                    updated_sentences[-1] = updated_sentences[-1] + " " + statement
                else:
                    updated_sentences.append(statement)

            else:
                updated_sentences.append("[replace_{}]".format(s_index))
                post_hoc_sentence["[replace_{}]".format(s_index)] = statement

        for s in list(post_hoc_sentence.values()):
            input_query = instructions.posthoc_attributions_paragraph_all.format_map(
                {"statement": s, "passages": passages}
            )

            if llama3_chat is True:
                input_query = create_prompt_with_llama3_format(input_query)

            prompts.append(input_query)

        outputs = []
        for input_query in prompts:
            raw_output = chat_completion(input_query, model=self.client_llm)
            outputs.append(raw_output)

        # process_output
        for output, sentence_key in zip(outputs, list(post_hoc_sentence.keys())):
            if (
                len(
                    [
                        t.split("[Response_End]")[0]
                        for t in output.split("[Response_Start]")
                        if "[Response_End]" in t
                    ]
                )
                == 0
            ):
                post_hoc_sentence[sentence_key] = post_hoc_sentence[sentence_key]
            else:
                processed_output = [
                    t.split("[Response_End]")[0]
                    for t in output.split("[Response_Start]")
                    if "[Response_End]" in t
                ][0]
                post_hoc_sentence[sentence_key] = processed_output

        final_processed_outputs = []
        for item in updated_sentences:
            if item in post_hoc_sentence:
                final_processed_outputs.append(post_hoc_sentence[item])
            else:
                final_processed_outputs.append(item)
        updated_sentences = final_processed_outputs

        return "\n".join(updated_sentences)

    def run(
        self,
        item,
        ranking_ce=False,
        use_feedback=False,
        skip_generation=False,
        posthoc_at=False,
        llama3_chat=False,
        task_name="default",
        zero_shot=False,
        max_per_paper=None,
        use_abstract=False,
        max_tokens=3000,
    ):
        # print("llama3 chat format? {0}".format(llama3_chat), flush=True)
        # print("use feedback: {}".format(use_feedback), flush=True)

        if ranking_ce is True:
            item["ctxs"], ranked_results, id_mapping = (
                self.reranking_passages_cross_encoder(item, use_abstract=False)
            )
            item["ranked_results"] = ranked_results
            item["id_mapping"] = id_mapping

        if max_per_paper is not None:
            filtered_ctxs = []
            title_to_count = {}
            for ctx in item["ctxs"]:
                if "title" not in ctx or ctx["title"] is None:
                    ctx["title"] = ""
                title_to_count.setdefault(ctx["title"], 0)
                if title_to_count[ctx["title"]] > max_per_paper:
                    # print("We have already aded the paper {0} {1} times".format(ctx["title"], max_per_paper))
                    continue
                else:
                    filtered_ctxs.append(ctx)
                    title_to_count[ctx["title"]] += 1

            item["ctxs"] = filtered_ctxs

        if skip_generation is False:
            generated_result, passages = self.generate_response(
                item,
                max_tokens=max_tokens,
                llama3_chat=llama3_chat,
                task_name=task_name,
                zero_shot=zero_shot,
            )
            if "\n\n References":
                generated_result = generated_result.split("\n\n References")[0]
            item["initial_result"] = generated_result

        if use_feedback is True:
            # print("[OpenScholar.run] Generating feedback...", flush=True)
            feedbacks = self.get_feedback(item, llama3_chat=llama3_chat)[:3]

            item["feedbacks"] = feedbacks
            with tqdm(total=len(feedbacks[:3]), desc="Processing feedbacks") as pbar:
                for feedback_idx, feedback in enumerate(feedbacks[:3]):
                    pbar.update(1)
                    # currently only supports non retrieval feedback
                    if len(feedback[1]) == 0:
                        edited_answer = self.edit_with_feedback(
                            item, feedback[0], llama3_chat=llama3_chat
                        )
                        if "Here is the revised answer:\n\n" in edited_answer:
                            edited_answer = edited_answer.split(
                                "Here is the revised answer:\n\n"
                            )[1]

                        if (
                            len(item["output"]) > 0
                            and len(edited_answer) / len(item["output"]) > 0.9
                        ):
                            item["output"] = edited_answer
                            item["edited_answer_{}".format(feedback_idx)] = (
                                edited_answer
                            )
                        else:
                            print("skipping as edited answers got too short")
                    else:
                        new_papers = []
                        if self.ss_retriever is True:
                            new_keywords = self.retrieve_keywords(feedback[1])
                            paper_list = {}
                            if len(new_keywords) > 0:
                                # print(f"\n[Used Feedback] Relevant keywords: {new_keywords}", flush=True)
                                for keyword in new_keywords:
                                    top_papers = search_paper_via_query(keyword)
                                    if top_papers is None:
                                        # print(f"Keywords: {keyword} | No pub related to the current keyword.", flush=True)
                                        pass
                                    else:
                                        for paper in top_papers:
                                            if paper["paperId"] not in paper_list:
                                                paper["text"] = paper["abstract"]
                                                paper["citation_counts"] = paper[
                                                    "citationCount"
                                                ]
                                                paper_list[paper["paperId"]] = paper
                                new_papers += list(paper_list.values())
                                # remove duplicarted data
                        if len(new_papers) > 0:
                            # print("before deduplication: {}".format(len(new_papers)))
                            new_papers_dicts = {
                                paper["text"][:100] + paper["title"]: paper
                                for paper in new_papers
                                if paper is not None and type(paper["text"]) is str
                            }
                            new_papers = list(new_papers_dicts.values())
                            # print("after deduplication: {}".format(len(new_papers)))
                            # add new papers when and only when we have the new papers.
                            if len(new_papers) > 0:
                                new_passages_reranked, _, _ = (
                                    self.reranking_passages_cross_encoder_supplemental(
                                        item, new_papers
                                    )
                                )
                                passages_start_index = len(item["ctxs"])

                                edited_answer = self.edit_with_feedback_retrieval(
                                    item,
                                    feedback[0],
                                    new_passages_reranked,
                                    passages_start_index,
                                )

                                if (
                                    len(item["output"]) > 0
                                    and len(edited_answer) / len(item["output"]) > 0.9
                                ):
                                    item["ctxs"] += new_passages_reranked[: self.top_n]
                                    item["edited_answer_{}".format(feedback_idx)] = (
                                        edited_answer
                                    )
                                    item["output"] = edited_answer
                                    item["edited_answer_{}".format(feedback_idx)] = (
                                        edited_answer
                                    )
                                elif (
                                    len(item["output"]) == 0 and len(edited_answer) > 0
                                ):
                                    item["ctxs"] += new_passages_reranked[: self.top_n]
                                    item["edited_answer_{}".format(feedback_idx)] = (
                                        edited_answer
                                    )
                                    item["output"] = edited_answer
                                    item["edited_answer_{}".format(feedback_idx)] = (
                                        edited_answer
                                    )
                                else:
                                    print("skipping as edited answers got too short")

        if posthoc_at is True:
            # attributed_results = self.insert_attributions_posthoc(item, llama3_chat=llama3_chat)
            # attributed_results = self.insert_attributions_posthoc_paragraph(item, llama3_chat=llama3_chat)
            attributed_results = self.insert_attributions_posthoc_paragraph_all(
                item, llama3_chat=llama3_chat
            )
            item["output"] = attributed_results

        item["output"] = (
            item["output"].replace("[Response_Start]", "").replace("[Response_End]", "")
        )

        # print(item["output"])

        # if "\n### References" in item["output"]:
        #     item["output"] = item["output"].split("\n### References")[0]
        return item


def process_paragraph(text):
    text = text.replace("<cit.>", "")
    text = remove_citations(text)
    return text


def process_input_data(data, use_contexts=True):
    processed_data = []
    for item in data:
        if "answer" not in item:
            item["answer"] = ""
        if "input" not in item:
            if "question" in item:
                item["input"] = item["question"]
            if "query" in item:
                item["input"] = item["query"]

        new_ctxs = []
        if use_contexts is True:
            # normalize ctx format for different retrieval APIs
            for ctx in item["ctxs"]:
                if type(ctx) is list:
                    for c in ctx:
                        if type(c) is dict:
                            new_ctxs.append(c)
                if type(ctx) is dict:
                    new_ctxs.append(ctx)
            item["ctxs"] = new_ctxs

            # remove duplicated contexts
            processed_paras = []
            with tqdm(total=len(item["ctxs"])) as pbar:
                for ctx in item["ctxs"]:
                    pbar.update(1)
                    if "retrieval text" in ctx:
                        ctx["text"] = ctx["retrieval text"]
                    if ctx["text"] is None or len(ctx["text"]) == 0:
                        continue
                    if type(ctx["text"]) != str:
                        ctx["text"] = " ".join(ctx["text"]["contexts"])
                    ctx["text"] = process_paragraph(ctx["text"])
                    if "title" not in ctx:
                        ctx["title"] = ""
                    processed_paras.append(ctx)

            processed_paras_dicts = {
                paper["text"][:100] + paper["title"]: paper for paper in processed_paras
            }
            processed_paras = list(processed_paras_dicts.values())

            item["ctxs"] = processed_paras
            item["original_ctxs"] = processed_paras
        processed_data.append(item)
    return processed_data


#### Pubmed Search Tool ####


class PubMedAPI(PubMedAPIWrapper):
    """
    Wrapper around PubMed API.

    This wrapper will use the PubMed API to conduct searches and fetch
    document summaries. By default, it will return the document summaries
    of the top-k results of an input search.

    Parameters:
        top_k_results: number of the top-scored document used for the PubMed tool
        MAX_QUERY_LENGTH: maximum length of the query.
            Default is 300 characters.
        doc_content_chars_max: maximum length of the document content.
            Content will be truncated if it exceeds this length.
            Default is 2000 characters.
        max_retry: maximum number of retries for a request. Default is 5.
        sleep_time: time to wait between retries.
            Default is 0.2 seconds.
        email: email address to be used for the PubMed API.
    """

    base_url_esearch: str = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
    )
    base_url_efetch: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
    max_retry: int = 3
    sleep_time: float = 0.2

    # Default values for the parameters
    top_k_results: int = 3
    MAX_QUERY_LENGTH: int = 300
    PAPER_BASE_URL = "https://pubmed.ncbi.nlm.nih.gov/"
    doc_content_chars_max: int = 2000
    email: str = "your_email@example.com"

    def run(self, query: str) -> str:
        """
        Run PubMed search and get the article meta information.
        See https://www.ncbi.nlm.nih.gov/books/NBK25499/#chapter4.ESearch
        It uses only the most informative fields of article meta information.
        """

        try:
            # Retrieve the top-k results for the query
            docs = [
                f"Published: {result['Published']}\n"
                f"Title: {result['Title']}\n"
                f"Copyright Information: {result['Copyright Information']}\n"
                f"Abstract::\n{result['Abstract']}"
                for result in self.load(query[: self.MAX_QUERY_LENGTH])
            ]

            # Join the results and limit the character count
            return (
                "\n\n".join(docs)[: self.doc_content_chars_max]
                if docs
                else "No good PubMed Result was found"
            )
        except Exception as ex:
            return f"PubMed exception: {ex}"

    def lazy_load(self, query: str) -> Iterator[dict]:
        """
        Search PubMed for documents matching the query.
        Return an iterator of dictionaries containing the document metadata.
        """

        url = (
            self.base_url_esearch
            + "db=pubmed&term="
            + str({urllib.parse.quote(query)})
            + f"&retmode=json&retmax={self.top_k_results}&usehistory=y"
        )
        result = urllib.request.urlopen(url)
        text = result.read().decode("utf-8")
        json_text = json.loads(text)
        webenv = json_text["esearchresult"]["webenv"]
        for uid in json_text["esearchresult"]["idlist"]:
            yield self.retrieve_article(uid, webenv)

    def load(self, query: str) -> List[dict]:
        """
        Search PubMed for documents matching the query.
        Return a list of dictionaries containing the document metadata.
        """
        return list(self.lazy_load(query))

    def _dict2document(self, doc: dict) -> Document:
        summary = doc.pop("Abstract")
        return Document(page_content=summary, metadata=doc)

    def lazy_load_docs(self, query: str) -> Iterator[Document]:
        for d in self.lazy_load(query=query):
            yield self._dict2document(d)

    def load_docs(self, query: str) -> List[Document]:
        return list(self.lazy_load_docs(query=query))

    def retrieve_article(self, uid: str, webenv: str) -> dict:
        url = (
            self.base_url_efetch
            + "db=pubmed&retmode=xml&id="
            + uid
            + "&webenv="
            + webenv
        )

        retry = 0
        while True:
            try:
                result = urllib.request.urlopen(url)
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and retry < self.max_retry:
                    # Too Many Requests errors
                    # wait for an exponentially increasing amount of time
                    print(  # noqa: T201
                        f"Too Many Requests, "
                        f"waiting for {self.sleep_time:.2f} seconds...",
                        flush=True,
                    )
                    time.sleep(self.sleep_time)
                    self.sleep_time *= 2
                    retry += 1
                else:
                    raise e

        xml_text = result.read().decode("utf-8")
        text_dict = self.parse(xml_text)
        return self._parse_article(uid, text_dict)

    def _parse_article(self, uid: str, text_dict: dict) -> dict:
        try:
            ar = text_dict["PubmedArticleSet"]["PubmedArticle"]["MedlineCitation"][
                "Article"
            ]
        except KeyError:
            ar = text_dict["PubmedArticleSet"]["PubmedBookArticle"]["BookDocument"]
        abstract_text = ar.get("Abstract", {}).get("AbstractText", [])
        summaries = [
            f"{txt['@Label']}: {txt['#text']}"
            for txt in abstract_text
            if "#text" in txt and "@Label" in txt
        ]
        summary = (
            "\n".join(summaries)
            if summaries
            else (
                abstract_text
                if isinstance(abstract_text, str)
                else (
                    "\n".join(str(value) for value in abstract_text.values())
                    if isinstance(abstract_text, dict)
                    else "No abstract available"
                )
            )
        )
        a_d = ar.get("ArticleDate", {})
        pub_date = "-".join(
            [a_d.get("Year", ""), a_d.get("Month", ""), a_d.get("Day", "")]
        )

        return {
            "uid": uid,
            "Title": ar.get("ArticleTitle", ""),
            "Published": pub_date,
            "url": self.PAPER_BASE_URL + uid,
            "Abstract": summary,
        }


#### Search API ####

"""
Academic Literature Search API

A comprehensive toolkit for searching, filtering, and analyzing academic literature
using semantic search, LLM-based relevance assessment, and intelligent keyword extraction.

Key Components:
- KeywordExtractor: Intelligent query generation for academic databases
- LLMPaperJudge: LLM-powered relevance assessment with explanations
- SemanticScholarSearch: Search interface for Semantic Scholar API
- OpenScholarReasoning: End-to-end literature analysis pipeline
"""


# =============================================================================
# KEYWORD EXTRACTION
# =============================================================================


class KeywordExtractor:
    """Intelligent keyword extraction for academic literature search."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def extract_keywords(
        self,
        question: str,
        model_name: str = "gpt-4o",
        query_num: int = 4,
        platform: str = "semantic_scholar",
        max_retries: int = 3,
    ) -> List[str]:
        """
        Generate optimized search keywords for academic literature retrieval.

        Args:
            question: The research question to generate keywords for
            model_name: The LLM model to use for keyword generation
            query_num: Number of queries to return
            platform: Target platform ('semantic_scholar', 'openalex', or 'auto')
            max_retries: Maximum retries for API calls

        Returns:
            List of optimized search queries
        """
        if self.verbose:
            print(
                f"[KEYWORD_GEN] Generating {query_num} keywords for {platform} using {model_name}"
            )

        enhanced_question = self._enhance_question_for_platform(question, platform)

        for attempt in range(max_retries):
            try:
                keywords = chat_completion(
                    instructions.keyword_extraction_prompt.format_map(
                        {"question": enhanced_question}
                    ),
                    model=model_name,
                )

                queries = self._parse_keyword_response(keywords, query_num)

                if queries and len(queries) >= min(query_num, 2):
                    if self.verbose:
                        print(f"[KEYWORD_GEN] Generated {len(queries)} keywords")
                        print(f"[KEYWORD_GEN] Keywords: {queries}")
                    return queries
                else:
                    if self.verbose:
                        print(
                            f"[KEYWORD_GEN] WARNING: Insufficient keywords on attempt {attempt + 1}"
                        )

            except Exception as e:
                if self.verbose:
                    print(f"[KEYWORD_GEN] ERROR: Attempt {attempt + 1} failed - {e}")

            if attempt < max_retries - 1:
                if self.verbose:
                    print("[KEYWORD_GEN] Retrying keyword generation...")
                time.sleep(1)

        if self.verbose:
            print(
                "[KEYWORD_GEN] WARNING: All attempts failed, using fallback extraction"
            )
        return self._extract_fallback_keywords(question, query_num, model_name)

    def _enhance_question_for_platform(self, question: str, platform: str) -> str:
        """Enhance question with platform-specific optimization."""
        if platform == "openalex":
            return f"For OpenAlex academic database search: {question}"
        elif platform == "semantic_scholar":
            return f"For Semantic Scholar academic search: {question}"
        return question

    def _parse_keyword_response(self, response: str, target_num: int = 4) -> List[str]:
        """Parse the LLM response to extract search keywords with multiple strategies."""
        if not response:
            return []

        # Strategy 1: Look for [Response_Start] and [Response_End] markers
        if "[Response_Start]" in response and "[Response_End]" in response:
            start_idx = response.find("[Response_Start]") + len("[Response_Start]")
            end_idx = response.find("[Response_End]")
            keywords_text = response[start_idx:end_idx].strip()
        else:
            # Strategy 2: Look for comma-separated terms
            lines = response.strip().split("\n")
            keywords_text = ""
            for line in reversed(lines):
                if "," in line and len(line.split(",")) >= 2:
                    keywords_text = line.strip()
                    break

            # Strategy 3: Use the entire response
            if not keywords_text:
                keywords_text = response.strip()

        # Clean and split keywords
        queries = []
        if keywords_text:
            raw_queries = keywords_text.split(",")
            for query in raw_queries:
                cleaned = self._clean_keyword_query(query)
                if cleaned and len(cleaned.split()) <= 8:
                    queries.append(cleaned)

        return queries[:target_num]

    def _clean_keyword_query(self, query: str) -> Optional[str]:
        """Clean and normalize a single keyword query."""
        query = query.strip()

        # Remove common artifacts
        artifacts = ["[Response_Start]", "[Response_End]", "Search queries:", "Query:"]
        for artifact in artifacts:
            query = query.replace(artifact, "")

        # Remove leading numbers and bullets
        query = re.sub(r"^\d+\.\s*", "", query)
        query = re.sub(r"^[-•*]\s*", "", query)

        # Normalize whitespace
        query = " ".join(query.split())

        return query if query else None

    def _extract_fallback_keywords(
        self, question: str, num_queries: int = 4, model_name: str = "gpt-4o"
    ) -> List[str]:
        """Intelligent fallback keyword extraction using LLM-powered analysis."""
        if self.verbose:
            print("[KEYWORD_GEN] Using intelligent fallback extraction")

        try:
            # First attempt: Simple LLM call
            fallback_prompt = f"""
Extract {num_queries} search queries from this question for academic literature search.
Focus on key concepts, entities, and terms that would be most effective for finding relevant papers.

Question: {question}

Return {num_queries} search queries, one per line, without numbering or bullets:
"""

            response = chat_completion(fallback_prompt, model=model_name)

            lines = [
                line.strip() for line in response.strip().split("\n") if line.strip()
            ]
            queries = []

            for line in lines[:num_queries]:
                cleaned = line
                cleaned = re.sub(r"^\d+\.?\s*", "", cleaned)  # Remove numbering
                cleaned = re.sub(r"^[-•*]\s*", "", cleaned)  # Remove bullets
                cleaned = cleaned.strip("\"'")  # Remove quotes

                if cleaned and len(cleaned.split()) >= 2:
                    queries.append(cleaned)

            if len(queries) >= 2:
                if self.verbose:
                    print(
                        f"[KEYWORD_GEN] LLM fallback generated {len(queries)} queries"
                    )
                return queries[:num_queries]

        except Exception as e:
            if self.verbose:
                print(f"[KEYWORD_GEN] ERROR: LLM fallback failed: {e}")

        # Rule-based fallback
        return self._rule_based_keyword_extraction(question, num_queries, model_name)

    def _rule_based_keyword_extraction(
        self, question: str, num_queries: int = 4, model_name: str = "gpt-4o"
    ) -> List[str]:
        """
        LLM-powered keyword extraction as ultimate fallback.
        Uses intelligent analysis without any hardcoded patterns.
        """
        if self.verbose:
            print("[KEYWORD_GEN] Using LLM-powered rule-based extraction")

        try:
            # Use LLM to intelligently analyze and extract keywords
            analysis_prompt = f"""
Analyze this research question and extract {num_queries} distinct search queries for academic literature databases.

Research Question: {question}

Instructions:
1. Identify the most important concepts, entities, and technical terms
2. Generate {num_queries} different search queries that would find relevant academic papers
3. Each query should focus on different aspects: specific terms, broader concepts, methodological approaches, related fields
4. Avoid common stop words and focus on meaningful academic terminology
5. Keep queries concise (2-6 words each)

Return exactly {num_queries} search queries, one per line, without numbering or formatting:
"""

            response = chat_completion(analysis_prompt, model=model_name)

            # Parse and clean the response
            lines = [
                line.strip() for line in response.strip().split("\n") if line.strip()
            ]
            queries = []

            for line in lines[:num_queries]:
                # Clean formatting artifacts
                cleaned = line
                cleaned = re.sub(r"^\d+\.?\s*", "", cleaned)  # Remove numbering
                cleaned = re.sub(r"^[-•*]\s*", "", cleaned)  # Remove bullets
                cleaned = cleaned.strip("\"'()[]{}")  # Remove wrapper chars
                cleaned = " ".join(cleaned.split())  # Normalize whitespace

                if cleaned and len(cleaned.split()) >= 2 and len(cleaned) >= 8:
                    queries.append(cleaned)

            if len(queries) >= 2:
                if self.verbose:
                    print(
                        f"[KEYWORD_GEN] LLM rule-based generated {len(queries)} queries"
                    )
                return queries[:num_queries]

        except Exception as e:
            if self.verbose:
                print(f"[KEYWORD_GEN] ERROR: LLM rule-based extraction failed: {e}")

        # Text-based fallback only if LLM completely fails
        return self._text_analysis_fallback(question, num_queries)

    def _text_analysis_fallback(self, question: str, num_queries: int = 4) -> List[str]:
        """
        Pure text analysis fallback using statistical methods only.
        No hardcoded patterns - purely data-driven.
        """
        if self.verbose:
            print("[KEYWORD_GEN] Using statistical text analysis fallback")

        # Tokenize
        words = re.findall(r"\b\w+\b", question.lower())
        word_freq = Counter(words)
        total_words = len(words)

        if not words:
            return [question[:50]]

        # Statistical analysis for stop word detection
        # Words that are very short or very frequent are likely stop words
        stop_words = set()
        for word, freq in word_freq.items():
            # Statistical thresholds based on distribution
            if (
                len(word) <= 2
                or freq / total_words > 0.25  # Very frequent
                or freq > total_words * 0.3
            ):  # Appears too often
                stop_words.add(word)

        # Extract meaningful terms using statistical measures
        meaningful_terms = []

        # Terms with intermediate frequency (not too rare, not too common)
        for word, freq in word_freq.items():
            if (
                word not in stop_words
                and len(word) > 3
                and 0.1 <= freq / total_words <= 0.8
            ):  # Sweet spot frequency
                meaningful_terms.append(word)

        # Add unique terms (appear only once but are long enough)
        unique_terms = [
            word
            for word, freq in word_freq.items()
            if freq == 1 and len(word) > 4 and word not in stop_words
        ]
        meaningful_terms.extend(unique_terms)

        # Add capitalized words from original question (likely entities)
        capitalized = re.findall(r"\b[A-Z][a-zA-Z]+\b", question)
        meaningful_terms.extend([term.lower() for term in capitalized])

        # Remove duplicates and sort by relevance
        meaningful_terms = list(set(meaningful_terms))

        # Create combinations
        queries = []
        if len(meaningful_terms) >= 2:
            # Strategy 1: First few meaningful terms
            queries.append(" ".join(meaningful_terms[:3]))

            # Strategy 2: Mix different parts
            if len(meaningful_terms) >= 4:
                queries.append(" ".join(meaningful_terms[1:4]))

            # Strategy 3: Longer combinations for specificity
            if len(meaningful_terms) >= 6:
                queries.append(" ".join(meaningful_terms[2:5]))

            # Strategy 4: Alternative combination
            if len(meaningful_terms) >= 5:
                queries.append(
                    " ".join(
                        [
                            meaningful_terms[0],
                            meaningful_terms[-1],
                            meaningful_terms[len(meaningful_terms) // 2],
                        ]
                    )
                )

        # Clean and validate
        final_queries = []
        seen = set()

        for query in queries:
            cleaned = " ".join(query.split())
            if (
                cleaned
                and cleaned not in seen
                and len(cleaned.split()) >= 2
                and len(cleaned) >= 8
            ):
                final_queries.append(cleaned)
                seen.add(cleaned)

        # If we still don't have enough, use sentence chunks
        if len(final_queries) < num_queries:
            # Split by punctuation and use meaningful chunks
            chunks = re.split(r"[.!?;,]", question)
            for chunk in chunks:
                chunk_words = [w for w in chunk.lower().split() if len(w) > 3]
                if len(chunk_words) >= 2:
                    chunk_query = " ".join(chunk_words[:4])
                    if chunk_query not in seen and len(chunk_query) >= 8:
                        final_queries.append(chunk_query)
                        seen.add(chunk_query)
                        if len(final_queries) >= num_queries:
                            break

        # Ultimate fallback
        if not final_queries:
            final_queries = [question[:50]]

        result = final_queries[:num_queries]
        if self.verbose:
            print(f"[KEYWORD_GEN] Statistical analysis generated {len(result)} queries")
        return result


# =============================================================================
# PAPER RELEVANCE ASSESSMENT
# =============================================================================


class LLMPaperJudge:
    """LLM-powered paper relevance assessment with detailed explanations."""

    def __init__(self):
        pass

    def judge_paper(
        self,
        query: str,
        paper_entity: Dict[str, Any],
        model_name: str,
        max_retries: int = 3,
        backoff_sec: float = 1.0,
        verbose: bool = True,
    ) -> Tuple[bool, str]:
        """
            Assess paper relevance and return decision with explanation.

        Args:
            query: The user's research question
                paper_entity: Dictionary containing paper metadata
                model_name: Name of the LLM model to use
                max_retries: Maximum number of retry attempts
                backoff_sec: Initial backoff time for exponential backoff

            Returns:
                Tuple of (relevance_decision, explanation_100_chars)
        """
        # Validate inputs
        if not self._validate_inputs(query, paper_entity, verbose):
            return False, "Invalid input data provided"

        # Prepare clean paper data
        try:
            clean_paper, paper_json_str = self._prepare_paper_data(paper_entity)
        except Exception as e:
            if verbose:
                print(f"[JUDGE_PAPER] ERROR: Error preparing paper data: {e}")
            return False, f"Data preparation error: {str(e)[:50]}"

        # Create enhanced prompt
        enhanced_prompt = self._create_enhanced_prompt(query, paper_json_str)

        # Retry logic with exponential backoff
        for attempt in range(1, max_retries + 1):
            try:
                # print(f"[JUDGE_PAPER] Attempt {attempt}/{max_retries}")

                response = chat_completion(enhanced_prompt, model=model_name).strip()
                decision, explanation = self._parse_judge_response(response)

                if decision is not None:
                    if verbose:
                        paper_title = clean_paper.get("title", "Unknown")[:50]
                        print(f"[JUDGE_PAPER] Decision: {decision}")
                        print(f"[JUDGE_PAPER] Explanation: {explanation}")
                        print(f"[JUDGE_PAPER] Paper: '{paper_title}...'")
                    return decision, explanation

                if verbose:
                    print(
                        f"[JUDGE_PAPER] WARNING: Could not parse response on attempt {attempt}"
                    )

                if attempt < max_retries:
                    wait_time = backoff_sec * (2 ** (attempt - 1))
                    time.sleep(wait_time)

            except Exception as e:
                if verbose:
                    print(f"[JUDGE_PAPER] ERROR: Attempt {attempt} failed: {e}")

                if attempt < max_retries:
                    wait_time = backoff_sec * (2 ** (attempt - 1))
                    time.sleep(wait_time)

        # All retries failed
        paper_title = clean_paper.get("title", "Unknown paper")[:40]
        default_explanation = (
            f"Failed assessment after {max_retries} attempts - defaulting to reject"
        )

        if verbose:
            print(
                f"[JUDGE_PAPER] WARNING: All {max_retries} attempts failed, defaulting to False"
            )
        return False, default_explanation

    def _validate_inputs(
        self, query: str, paper_entity: Dict[str, Any], verbose: bool = True
    ) -> bool:
        """Validate input parameters."""
        if not query or not isinstance(query, str):
            if verbose:
                print(f"[JUDGE_PAPER] ERROR: Invalid query: {query}")
            return False

        if not paper_entity or not isinstance(paper_entity, dict):
            if verbose:
                print(f"[JUDGE_PAPER] ERROR: Invalid paper entity: {paper_entity}")
            return False

        return True

    def _prepare_paper_data(
        self, paper_entity: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], str]:
        """
        Prepare clean paper data for the prompt, including main text if available.

        Includes: title, abstract, main text/passages, year, authors, url
        """
        clean_paper = {
            "title": (paper_entity.get("title") or "").strip(),
            "abstract": (paper_entity.get("abstract") or "").strip(),
            "year": paper_entity.get("year", ""),
            "authors": paper_entity.get("authors", []),
            "url": paper_entity.get("url", ""),
        }

        # Include main text/passages if available (for more comprehensive assessment)
        # Check various possible fields for text content
        text_fields = ["text", "full_text", "passages", "content", "body"]
        for field in text_fields:
            if field in paper_entity and paper_entity[field]:
                content = paper_entity[field]

                # Handle list of passages vs single text
                if isinstance(content, list):
                    # Take first few passages (intro) and last few (conclusion)
                    if len(content) > 10:
                        # Take first 5 and last 5 paragraphs
                        selected = content[:5] + content[-5:]
                        clean_paper["key_sections"] = " ".join(
                            p if isinstance(p, str) else p.get("text", "")
                            for p in selected
                        )[
                            :3000
                        ]  # Limit to 3000 chars
                    else:
                        clean_paper["main_text"] = " ".join(
                            p if isinstance(p, str) else p.get("text", "")
                            for p in content
                        )[
                            :3000
                        ]  # Limit to 3000 chars
                elif isinstance(content, str):
                    # Truncate long text but keep beginning and end
                    if len(content) > 3000:
                        clean_paper["main_text"] = (
                            content[:1500] + "\n...\n" + content[-1500:]
                        )
                    else:
                        clean_paper["main_text"] = content
                break  # Only use first available text field

        # Remove empty fields
        clean_paper = {k: v for k, v in clean_paper.items() if v}
        paper_json_str = json.dumps(clean_paper, indent=2)

        return clean_paper, paper_json_str

    def _create_enhanced_prompt(self, query: str, paper_json_str: str) -> str:
        """Create enhanced prompt for detailed feedback."""
        return f"""
{instructions.relevance_judge_prompt.format_map({
            "query": query.strip(),
            "paper_json": paper_json_str
})}

Additionally, provide a brief explanation (approximately 100 characters) for your decision.

Response format:
Decision: [True/False]
Explanation: [Your ~100 character explanation]
"""

    def _parse_judge_response(self, response: str) -> Tuple[Optional[bool], str]:
        """Parse the LLM judge response to extract decision and explanation."""
        # Strategy 1: Structured format
        decision_match = re.search(r"Decision:\s*(True|False)", response, re.IGNORECASE)
        explanation_match = re.search(
            r"Explanation:\s*(.+?)(?:\n|$)", response, re.IGNORECASE
        )

        if decision_match and explanation_match:
            decision = decision_match.group(1).lower() == "true"
            explanation = explanation_match.group(1).strip()[:100]
            return decision, explanation

        # Strategy 2: Simple True/False with explanation
        simple_match = re.search(
            r"(True|False)\s*[:\-]?\s*(.+)", response, re.IGNORECASE
        )
        if simple_match:
            decision = simple_match.group(1).lower() == "true"
            explanation = simple_match.group(2).strip()[:100]
            return decision, explanation

        # Strategy 3: Just True/False
        if response.strip() in {"True", "False"}:
            decision = response.strip() == "True"
            explanation = (
                "Accept paper as relevant"
                if decision
                else "Reject paper as not relevant"
            )
            return decision, explanation

        # Strategy 4: Contains True or False
        if "True" in response and "False" not in response:
            explanation = (
                response.strip()[:100]
                if response.strip()
                else "Accept paper as relevant"
            )
            return True, explanation
        elif "False" in response and "True" not in response:
            explanation = (
                response.strip()[:100]
                if response.strip()
                else "Reject paper as not relevant"
            )
            return False, explanation

        return None, f"Could not parse response: {response[:50]}..."


class PaperQueryAligner:
    """Filter papers using parallel LLM relevance assessment."""

    def __init__(self, judge: LLMPaperJudge = None):
        self.judge = judge or LLMPaperJudge()

    def filter_papers(
        self,
        query: str,
        paper_list: Iterable[Dict[str, Any]],
        model_name: str = "o1-mini-2025-01",
        max_workers: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Filter papers in parallel using LLM relevance assessment.

        Args:
            query: The user query
            paper_list: Papers to evaluate
            model_name: LLM model identifier
            max_workers: Degree of parallelism

        Returns:
            Subset of paper_list deemed relevant
        """
        results = [None] * len(paper_list)

        def _wrapper(idx: int, paper: Dict[str, Any]):
            is_relevant, explanation = self.judge.judge_paper(query, paper, model_name)

            # Log the assessment with 100-character explanation
            paper_title = paper.get("title", "Unknown")[:30]
            status = "ACCEPT" if is_relevant else "REJECT"
            print(f"[PAPER_JUDGE] {status}: {paper_title}... | {explanation}")

            return idx, is_relevant

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_wrapper, i, p) for i, p in enumerate(paper_list)
            ]
            for fut in as_completed(futures):
                idx, is_relevant = fut.result()
                results[idx] = is_relevant

        return [paper for paper, flag in zip(paper_list, results) if flag]


# =============================================================================
# SEARCH FUNCTIONALITY
# =============================================================================


class SemanticScholarSearch:
    """Search interface for Semantic Scholar API."""

    def __init__(self, extractor: KeywordExtractor = None, verbose: bool = True):
        self.extractor = extractor or KeywordExtractor()
        self.verbose = verbose

    def search(
        self,
        question: str,
        model_name: str,
        max_paper_num: int = 2,
        attempt: int = 3,
        min_citation_count: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Search Semantic Scholar for relevant papers.

        Args:
            question: Research question
            model_name: LLM model for keyword generation
            max_paper_num: Maximum papers to retrieve
            attempt: Number of search attempts
            min_citation_count: Minimum citation count filter

        Returns:
            List of processed paper dictionaries
        """
        new_keywords = self.extractor.extract_keywords(
            question, model_name=model_name, platform="semantic_scholar"
        )

        if self.verbose:
            print(f"[SEMANTIC_SCHOLAR] Using keywords: {new_keywords}")

        paper_list = {}
        successful_queries = 0

        for i, keyword in enumerate(new_keywords):
            if self.verbose:
                print(
                    f"[SEMANTIC_SCHOLAR] Processing keyword {i+1}/{len(new_keywords)}: '{keyword}'"
                )

            try:
                top_papers = search_paper_via_query(
                    query=keyword,
                    max_paper_num=max_paper_num,
                    attempt=attempt,
                    minCitationCount=min_citation_count,
                )

                if top_papers is None:
                    if self.verbose:
                        print(
                            f"[SEMANTIC_SCHOLAR] WARNING: No papers found for keyword: '{keyword}'"
                        )
                    continue

                successful_queries += 1
                if self.verbose:
                    print(
                        f"[SEMANTIC_SCHOLAR] Found {len(top_papers)} papers for keyword: '{keyword}'"
                    )

                for paper in top_papers:
                    if paper["paperId"] not in paper_list:
                        paper["text"] = paper["abstract"]
                        paper["citation_counts"] = paper["citationCount"]
                        paper_list[paper["paperId"]] = paper

            except Exception as e:
                if self.verbose:
                    print(
                        f"[SEMANTIC_SCHOLAR] ERROR: Search failed for '{keyword}': {e}"
                    )

        if successful_queries == 0:
            if self.verbose:
                print("[SEMANTIC_SCHOLAR] ERROR: No successful queries found")
            return []

        if self.verbose:
            print(
                f"[SEMANTIC_SCHOLAR] {successful_queries}/{len(new_keywords)} queries successful, {len(paper_list)} unique papers found"
            )

        # Process papers into final format
        final_paper_list = []
        for paper_id in paper_list:
            paper = paper_list[paper_id]

            # Add abstract entry
            final_paper_list.append(
                {
                    "semantic_scholar_id": paper_id,
                    "type": "ss_abstract",
                    "year": paper["year"],
                    "authors": paper["authors"],
                    "title": paper["title"],
                    "text": paper["text"],
                    "url": paper["url"],
                    "citation_counts": paper["citationCount"],
                    "abstract": paper["abstract"],
                }
            )

            # Add ArXiv passages if available
            if paper.get("externalIds") is not None and "ArXiv" in paper["externalIds"]:
                try:
                    passages = ArxivPaperProcessor.retrieve_passages(
                        paper["externalIds"]["ArXiv"]
                    )
                    for passage in passages:
                        final_paper_list.append(
                            {
                                "semantic_scholar_id": paper_id,
                                "type": "ss_passage",
                                "year": paper["year"],
                                "authors": paper["authors"],
                                "title": paper["title"],
                                "text": passage,
                                "url": paper["url"],
                                "citation_counts": paper["citationCount"],
                                "abstract": paper["abstract"],
                            }
                        )
                except Exception as e:
                    if self.verbose:
                        print(
                            f"[SEMANTIC_SCHOLAR] WARNING: Failed to retrieve ArXiv passages for {paper_id}: {e}"
                        )

        return final_paper_list


class ArxivPaperProcessor:
    """Process papers from ArXiv for text extraction."""

    @staticmethod
    def retrieve_passages(arxiv_id: str) -> List[str]:
        """Retrieve passages from a single ArXiv paper."""
        ar5iv_link = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
        return ArxivPaperProcessor._parse_paragraphs(ar5iv_link)

    @staticmethod
    def _parse_paragraphs(link: str) -> List[str]:
        """Parse paragraphs from ArXiv HTML."""
        response = requests.get(link, verify=False)
        time.sleep(0.1)

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract abstract
        raw_abstract = soup.find_all("div", "ltx_abstract")
        try:
            abstract = "".join(raw_abstract[0].text.split("\n")[2:])
        except:
            abstract = ""

        # Extract sections
        subsections = soup.find_all(class_="ltx_para", id=re.compile(r"^S\d+\.+(p|S)"))

        paragraphs = []
        for subsection in subsections:
            paragraphs.append(re.sub(r"\n", "", subsection.text))

        return paragraphs


# =============================================================================
# TEXT PROCESSING UTILITIES
# =============================================================================


class TextProcessor:
    """Utilities for text processing and cleaning."""

    @staticmethod
    def remove_citations(text: str) -> str:
        """Remove citation markers from text."""
        text = re.sub(r"\[\d+", "", text)
        text = re.sub(r" \[\d+", "", text)
        return text.replace(" |", "").replace("]", "")

    @staticmethod
    def process_paragraph(text: str) -> str:
        """Clean and process paragraph text."""
        text = text.replace("<cit.>", "")
        return TextProcessor.remove_citations(text)

    @staticmethod
    def process_input_data(data: List[Dict], use_contexts: bool = False) -> List[Dict]:
        """Process input data for OpenScholar reasoning."""
        processed_data = []

        for item in data:
            # Ensure required fields
            if "answer" not in item:
                item["answer"] = ""
            if "input" not in item:
                if "question" in item:
                    item["input"] = item["question"]
                elif "query" in item:
                    item["input"] = item["query"]

            if use_contexts:
                item = TextProcessor._process_contexts(item)

            processed_data.append(item)

        return processed_data

    @staticmethod
    def _process_contexts(item: Dict) -> Dict:
        """Process context data for an item."""
        new_ctxs = []

        # Normalize context format
        for ctx in item.get("ctxs", []):
            if isinstance(ctx, list):
                new_ctxs.extend([c for c in ctx if isinstance(c, dict)])
            elif isinstance(ctx, dict):
                new_ctxs.append(ctx)

        item["ctxs"] = new_ctxs

        # Remove duplicated contexts
        processed_paras = []

        with tqdm(total=len(item["ctxs"])) as pbar:
            for ctx in item["ctxs"]:
                pbar.update(1)

                if "retrieval text" in ctx:
                    ctx["text"] = ctx["retrieval text"]

                if not ctx.get("text"):
                    continue

                if not isinstance(ctx["text"], str):
                    ctx["text"] = " ".join(ctx["text"]["contexts"])

                ctx["text"] = TextProcessor.process_paragraph(ctx["text"])

                if "title" not in ctx:
                    ctx["title"] = ""

                processed_paras.append(ctx)

        # Deduplicate by text + title
        processed_paras_dict = {
            paper["text"][:100] + paper["title"]: paper for paper in processed_paras
        }

        item["ctxs"] = list(processed_paras_dict.values())
        item["original_ctxs"] = item["ctxs"]

        return item


# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================


class OpenScholarReasoning:
    """End-to-end literature analysis pipeline."""

    def __init__(self):
        self.extractor = KeywordExtractor()
        self.judge = LLMPaperJudge()
        self.aligner = PaperQueryAligner(self.judge)
        self.searcher = SemanticScholarSearch(self.extractor)

    def reason(
        self,
        query: str,
        model_name: str = "gpt-4o",
        default_reranker: str = "OpenSciLM/OpenScholar_Reranker",
        **kwargs,
    ) -> Tuple[str, Optional[Any]]:
        """
        Perform end-to-end literature reasoning.

        Args:
            query: Research question
            model_name: LLM model name
            default_reranker: Reranker model name
            **kwargs: Additional parameters

        Returns:
            Tuple of (answer, metadata)
        """
        # Search for papers
        ss_retrieved_passages = self.searcher.search(
            question=query,
            model_name=model_name,
            max_paper_num=kwargs.get("max_paper_num", 5),
            min_citation_count=kwargs.get("min_citation", 0),
        )

        # Filter relevant papers with adaptive approach
        relevant_papers = self.aligner.filter_papers(
            query=query,
            paper_list=ss_retrieved_passages,
            model_name=model_name,
        )

        # Adaptive fallback: if no papers found, try more relaxed approach for complex queries
        if not relevant_papers and ss_retrieved_passages:
            print(
                f"[ADAPTIVE_FILTER] No papers passed initial filter. Trying exploratory approach for {len(ss_retrieved_passages)} papers..."
            )

            # Create a more exploratory prompt for complex/novel queries
            exploratory_aligner = PaperQueryAligner(judge=LLMPaperJudge())

            # Use a special flag to indicate exploratory mode
            exploratory_query = f"[EXPLORATORY_MODE] {query}"
            relevant_papers = exploratory_aligner.filter_papers(
                query=exploratory_query,
                paper_list=ss_retrieved_passages,
                model_name=model_name,
            )

            if relevant_papers:
                print(
                    f"[ADAPTIVE_FILTER] Exploratory approach found {len(relevant_papers)} potentially relevant papers"
                )
            else:
                print(
                    f"[ADAPTIVE_FILTER] Even exploratory approach found no relevant papers"
                )

        if not relevant_papers:
            return (
                "ABSTAIN — no relevant scientific literature available for this query",
                None,
            )

        # Process data
        reason_dict = [{"input": query, "ctxs": relevant_papers}]

        if not relevant_papers:
            kwargs["ranking_ce"] = False
            kwargs["use_contexts"] = False
            kwargs["max_per_paper"] = None

        # Initialize reranker and OpenScholar
        print(
            f"[OpenScholarReasoning] Initializing reranker: {default_reranker}",
            flush=True,
        )
        print(
            f"[OpenScholarReasoning] First-time use may download model files (~1GB)...",
            flush=True,
        )

        reranker = FlagReranker(default_reranker, use_fp16=True)

        print(f"[OpenScholarReasoning] Reranker initialized successfully", flush=True)

        open_scholar = OpenScholar(
            model=kwargs.get("model"),
            tokenizer=kwargs.get("tokenizer"),
            client_llm=model_name,
            reranker=reranker,
            use_reranker=True,
            query_type="citation_qa",
        )

        # Process and generate response
        final_results = open_scholar.generate(
            reason_dict,
            top_n=kwargs.get("top_n", 4),
            min_citation=kwargs.get("min_citation", 4),
            norm_cite=kwargs.get("norm_cite", True),
            ss_retriever=kwargs.get("ss_retriever", True),
            use_contexts=kwargs.get("use_contexts", True),
            ranking_ce=kwargs.get("ranking_ce", True),
            feedback=kwargs.get("feedback", True),
            skip_generation=kwargs.get("skip_generation", False),
            posthoc_at=kwargs.get("posthoc_at", True),
            zero_shot=kwargs.get("zero_shot", True),
            use_abstract=kwargs.get("use_abstract", True),
            max_per_paper=kwargs.get("max_per_paper", 3),
            max_tokens=kwargs.get("max_tokens", 2000),
            task_name=kwargs.get("task_name", "default"),
        )

        return final_results[0], None


# =============================================================================
# LEGACY FUNCTION COMPATIBILITY
# =============================================================================

# Initialize global instances for backward compatibility
_extractor = KeywordExtractor()
_judge = LLMPaperJudge()
_aligner = PaperQueryAligner(_judge)
_searcher = SemanticScholarSearch(_extractor)
_reasoning = OpenScholarReasoning()


# Legacy function wrappers
def retrieve_keywords(
    question,
    model_name="gpt-4o",
    query_num=4,
    platform="semantic_scholar",
    max_retries=3,
    verbose=True,
):
    """Legacy wrapper for keyword extraction."""
    # Create a temporary extractor with the specified verbose setting
    extractor = KeywordExtractor(verbose=verbose)
    return extractor.extract_keywords(
        question, model_name, query_num, platform, max_retries
    )


def search_semantic_scholar(
    question, model_name, max_paper_num=2, attempt=3, minCitationCount=5
):
    """Legacy wrapper for Semantic Scholar search."""
    papers = _searcher.search(
        question, model_name, max_paper_num, attempt, minCitationCount
    )
    keywords = _extractor.extract_keywords(
        question, model_name, platform="semantic_scholar"
    )
    return papers, keywords


def paper_query_aligner(query, paper_list, model_name="o1-mini-2025-01", max_workers=4):
    """Legacy wrapper for paper filtering."""
    return _aligner.filter_papers(query, paper_list, model_name, max_workers)


def retrieve_passages_single_paper(arxiv_id):
    """Legacy wrapper for ArXiv paper processing."""
    return ArxivPaperProcessor.retrieve_passages(arxiv_id)


def parsing_paragraph(link):
    """Legacy wrapper for paragraph parsing."""
    return ArxivPaperProcessor._parse_paragraphs(link)


def remove_citations(sent):
    """Legacy wrapper for citation removal."""
    return TextProcessor.remove_citations(sent)


def process_paragraph(text):
    """Legacy wrapper for paragraph processing."""
    return TextProcessor.process_paragraph(text)


def process_input_data(data, use_contexts=False):
    """Legacy wrapper for input data processing."""
    return TextProcessor.process_input_data(data, use_contexts)


def openscholar_reasoning(query, **kwargs):
    """Legacy wrapper for OpenScholar reasoning."""
    return _reasoning.reason(query, **kwargs)
