#!/usr/bin/env python3
"""
Intelligent Selection - RAG-based routing and selection tool
Modular tool for making intelligent choices from any list
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Union, Dict
import json

# Import from embedding utils
from embedding_utils import (
    create_embeddings,
    vector_search,
    chunk_text_logical,
    load_llama_cpp_model,
)


def intelligent_selection(
    prompt: str,
    corpus: Union[List[str], str],
    top_k: int = 4,
    create_new: bool = False,
    selection_mode: str = "single",
    embedding_model: Optional[str] = None,
    llm=None,
) -> str:
    """
    Intelligent selection using RAG + LLM decision making

    Modes:
    - single: Select ONE item from corpus (e.g., choose tool)
    - classification: Classify into category or create new (e.g., Topic routing)

    Args:
        prompt: User's query/prompt
        corpus: List of items OR raw text to chunk
        top_k: Number of top items to retrieve (default: 4)
        create_new: If True, allow creating new choice (classification mode)
        selection_mode: "single" | "classification"
        embedding_model: Path to embedding model
        llm: LLM instance for final decision (optional, will create if None)

    Returns:
        Selected item or classification result
    """
    print(f"🧠 Intelligent Selection ({selection_mode} mode)")
    print(f"   Prompt: {prompt[:100]}...")

    # 1. Chunk corpus (logical 3-sentence chunks)
    if isinstance(corpus, str):
        print("   Chunking corpus (logical 3-sentence chunks)...")
        chunks = chunk_text_logical(corpus, chunk_size=3)
    else:
        chunks = corpus  # Already a list

    print(f"   Total chunks: {len(chunks)}")

    # 2. Get embedding model path
    if not embedding_model:
        from embedding_utils import get_default_embedding_model

        embedding_model = get_default_embedding_model()

    if not embedding_model:
        print("   ⚠  No embedding model found!")
        return ""

    print(f"   Embedding model: {Path(embedding_model).name}")

    # 3. Create embeddings
    print("   Creating embeddings...")
    embeddings = create_embeddings(chunks, embedding_model)

    # 4. Vector search to find top_k relevant chunks
    print(f"   Finding top {top_k} relevant items...")
    top_items, scores = vector_search(
        prompt, embeddings, chunks, top_k=top_k, model_path=embedding_model
    )

    print(f"   Top matches (scores):")
    for i, (item, score) in enumerate(zip(top_items, scores)):
        print(f"      {i + 1}. {item[:80]}... ({score:.3f})")

    # 5. Format for LLM decision
    context = "\n\n".join(
        [f"OPTION {i + 1}: {item}" for i, item in enumerate(top_items)]
    )

    # 6. Build LLM prompt with "create new" instruction
    create_new_text = ""
    if selection_mode == "single":
        final_instruction = (
            f"INSTRUCTION: Select the SINGLE best option from the {top_k} options above "
            f"to fulfill the user's request. "
            f"Respond ONLY with the exact text of the selected option."
        )
    elif selection_mode == "classification":
        if create_new:
            create_new_text = (
                "If none of the {top_k} options are relevant, you MUST create a new "
                f"category that better fits the user's input. "
                f"Use a clear, descriptive name for the new category."
            )
        else:
            create_new_text = ""

        final_instruction = (
            f"INSTRUCTION: Classify the user's input into ONE of the {top_k} categories above "
            f"OR create a new category if none are relevant. "
            f"{create_new_text} "
            f"Respond ONLY with the selected category name or new category name."
        )
    else:
        raise ValueError(f"Unknown selection_mode: {selection_mode}")

    llm_prompt = (
        f"USER REQUEST: {prompt}\n\n"
        f"--- TOP {top_k} MOST RELEVANT OPTIONS ---\n"
        f"{context}\n\n"
        f"--- {final_instruction}"
    )

    # 7. Load LLM (if not provided)
    if not llm:
        model_path = os.environ.get(
            "SENTER_TEXT_MODEL_PATH",
            "/home/sovthpaw/Models/Hermes-3-Llama-3.2-3B.Q4_K_M.gguf",
        )

        if not Path(model_path).exists():
            print(f"   ⚠ Model not found, using fallback...")
            model_path = "/home/sovthpaw/Models/Hermes-3-Llama-3.2-3B.Q4_K_M.gguf"

        if not Path(model_path).exists():
            print("   ⚠  No LLM model available!")
            return ""

        print(f"   Loading LLM: {Path(model_path).name}")
        llm = load_llama_cpp_model(model_path, embedding=False)

    # 8. LLM makes final decision
    if not llm:
        print("   ⚠  No LLM available, returning first option")
        return top_items[0] if top_items else ""

    print("   Making decision from top options...")
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": llm_prompt}],
        max_tokens=128,
        temperature=0.3,  # Low temp for deterministic selection
    )

    decision = response["choices"][0]["message"]["content"].strip()
    print(f"✅ Selected: {decision}")

    # 9. Check if new category was created
    if selection_mode == "classification" and create_new:
        is_existing = any(decision.lower() in item.lower() for item in top_items)
        if not is_existing:
            print(f"   🆕 Created new category: {decision}")

    return decision


# Convenience wrappers for common use cases
def select_tool(
    prompt: str, tools: List[str], embedding_model: str = None, llm=None
) -> str:
    """
    Select best tool from list

    Example use:
        tools = ["git", "npm", "docker", "kubectl"]
        selected = select_tool("I need to deploy a container", tools)
        # Returns: "docker"
    """
    return intelligent_selection(
        prompt=prompt,
        corpus=tools,
        top_k=4,
        create_new=False,
        selection_mode="single",
        embedding_model=embedding_model,
        llm=llm,
    )


def classify_topic(
    prompt: str,
    topics: List[str],
    create_new: bool = True,
    embedding_model: str = None,
    llm=None,
) -> str:
    """
    Classify content into Topic or create new

    Example use:
        topics = ["general", "coding", "creative", "research"]
        topic = classify_topic("I want to generate some music", topics)
        # Returns: "creative" or new topic like "music_generation"
    """
    return intelligent_selection(
        prompt=prompt,
        corpus=topics,
        top_k=4,
        create_new=create_new,
        selection_mode="classification",
        embedding_model=embedding_model,
        llm=llm,
    )


def select_agent(
    prompt: str,
    agents: List[dict],
    create_new: bool = True,
    embedding_model: str = None,
    llm=None,
) -> dict:
    """
    Select best agent from list of agent dicts

    Args:
        prompt: User request
        agents: List of agent dicts with 'id', 'name', 'description' keys
        create_new: If True, can return a new agent suggestion
        embedding_model: Path to embedding model
        llm: LLM instance

    Returns:
        Selected agent dict

    Example:
        agents = [
            {"id": "coding", "name": "Coding Agent", "description": "Handles programming tasks"},
            {"id": "creative", "name": "Creative Agent", "description": "Handles music/image generation"}
        ]
        selected = select_agent("I need to write a Python script", agents)
        # Returns: {"id": "coding", ...}
    """
    agent_names = [f"{a['id']}: {a['description']}" for a in agents]
    selected_id = intelligent_selection(
        prompt=prompt,
        corpus=agent_names,
        top_k=4,
        create_new=create_new,
        selection_mode="classification",
        embedding_model=embedding_model,
        llm=llm,
    )

    # Find and return agent dict
    for agent in agents:
        if selected_id.lower() == agent["id"].lower():
            return agent

    # If new agent was suggested
    return {
        "id": selected_id,
        "name": selected_id.replace("_", " ").title(),
        "description": f"New agent for: {prompt[:50]}...",
    }


def main():
    """CLI for testing intelligent_selection"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Intelligent Selection - RAG-based routing and selection"
    )
    parser.add_argument("--prompt", "-p", required=True, help="User prompt/query")
    parser.add_argument(
        "--mode",
        "-m",
        choices=["single", "classification"],
        default="single",
        help="Selection mode",
    )
    parser.add_argument("--items", "-i", nargs="+", help="List of items to choose from")
    parser.add_argument(
        "--create-new",
        "-n",
        action="store_true",
        help="Allow creating new item (classification mode only)",
    )
    parser.add_argument("--corpus-file", "-f", help="Text file to use as corpus")

    args = parser.parse_args()

    # Get corpus
    if args.items:
        corpus = args.items
    elif args.corpus_file:
        with open(args.corpus_file, "r") as f:
            corpus = f.read()
    else:
        print("Error: Provide --items or --corpus-file")
        return

    # Run selection
    result = intelligent_selection(
        prompt=args.prompt,
        corpus=corpus,
        top_k=4,
        create_new=args.create_new,
        selection_mode=args.mode,
    )

    print(f"\n📊 RESULT: {result}")


if __name__ == "__main__":
    main()
