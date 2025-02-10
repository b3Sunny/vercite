"""
RAG system prompt templates collection.

Contains various prompt templates for claim processing and analysis using LLMs.
"""

from langchain_core.prompts import ChatPromptTemplate

PREPROCESSING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that generates search queries for a RAG system. 
Your task is to generate search queries for a given claim and reference. 

**Guidelines:**
1. Extract the relevant part of the claim associated with the given reference and use it as the first query without changes, except to remove any brackets (e.g., "[15]").
2. Generate three additional queries that are distinct but retain the core meaning of the extracted part of the claim.
3. Consider the context to improve query relevance.
4. Format the output strictly in the following JSON structure:

Example claim: "To answer the research questions, we compared full finetuning with three popular PEFT methods: LoRA (Low-Rank Adaptation) [15], (IA) 3 (Infused Adapter by Inhibiting and Amplifying Inner Activations) [16], and prompt tuning [17]."
Example output:
{{
    "main_query": "(IA) 3 (Infused Adapter by Inhibiting and Amplifying Inner Activations)",
    "rewritten_queries": [
        "Example variant 1",
        "Example variant 2",
        "Example variant 3"
    ]
}}
"""),
    ("user", """Claim: {claim}
Context before claim: {context_before}
Context after claim: {context_after}
Cited Paper: {reference}""")
])

PROCESSING_SOURCE_ONLY_PROMPT = ChatPromptTemplate.from_template(
    """
    Below are several text excerpts from different PDFs that should support a given statement. 
    Your task is to select the text that best supports the statement. 
    If none of the texts sufficiently supports the statement, indicate that as well.

    Claim: {claim}

    Texts:
    {docs}

    Return ONLY the name of the PDF e.g. [2]_2309.012193 that provides the best support for the statement.
    If none of the texts sufficiently supports the statement, return 'none'.
    Return only the filename, nothing else.
    """
)

PROCESSING_DETAILED_SOURCE_PROMPT = ChatPromptTemplate.from_template(
    """
    Below are several text excerpts from different PDFs that should support a given statement. 
    Your task is to select the text that best supports the statement. 
    If none of the texts sufficiently supports the statement, indicate that as well.

    Claim: {claim}

    Texts:
    {docs}

    Return a JSON object with the following structure:
    {{
        "predicted_source": "PDF filename e.g. [2]_2309.012193 without file format",
        "page": "Page number where the evidence was found",
        "reason": "Brief explanation why this source was chosen",
        "relevant_text": "The specific text excerpt that supports the claim",
        "confidence": "High/Medium/Low"
    }}

    If no suitable evidence is found, return:
    {{
        "predicted_source": "none",
        "page": "",
        "reason": "No supporting evidence found",
        "relevant_text": "",
        "confidence": "None"
    }}
    """
)

# Old prompts

DETAILED_PROMPT = ChatPromptTemplate.from_template(
    """
    Below are several text excerpts from different PDFs that should support a given statement. 
    Your task is to select the text that best supports the statement. 
    If none of the texts sufficiently supports the statement, indicate that as well.

    Claim: {claim}

    Texts:

    {docs}

    Return the name of the PDF e.g. [2]_2309.012193 and the Page that provides the best support for the statement. 
    Give maximum 3 sources, if it makes sense.
    Give justification for your choice and veracity score of the source.
    If none of the texts sufficiently supports the statement, return 'No suitable evidence found.'
    """
)

SIMPLE_PROMPT = ChatPromptTemplate.from_template(
    """
    Below are several text excerpts from different PDFs that should support a given statement. 
    Your task is to select the text that best supports the statement. 
    If none of the texts sufficiently supports the statement, indicate that as well.

    Claim: {claim}

    Texts:

    {docs}

    Return the name of the PDF e.g. [2]_2309.012193 and the Page that provides the best support for the statement.
    """
)
