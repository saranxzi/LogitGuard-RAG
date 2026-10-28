"""
LogitGuard-RAG: 100+ Question Benchmark Suite Builder
Compiles a balanced, rigorously categorized benchmark of 110 test questions spanning:
  1. RAGTruth Multi-Task QA/Summary/Data2txt (50 items: 32 with human gold spans, 18 clean)
  2. Contrastive Factual Perturbations (30 items: 15 clean orig, 15 perturbed conflicts)
  3. Open-Domain Authoritative Inquiries (20 items: history, science, tech origins)
  4. Scaffolding & Formatting Structural Keys (10 items: headers, colon keys, metadata)
"""

import json
from pathlib import Path
from typing import List, Dict, Any

from logitguard.config import PROJECT_ROOT


def build_benchmark_100() -> Path:
    out_file = PROJECT_ROOT / "data" / "benchmark_100.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []

    # ---------------------------------------------------------------------------
    # 1. RAGTruth Multi-Task (50 items)
    # ---------------------------------------------------------------------------
    demo_cache_path = PROJECT_ROOT / "data" / "cached_demo_samples.json"
    if demo_cache_path.exists():
        with open(demo_cache_path, "r", encoding="utf-8") as f:
            demo_items = json.load(f)
            for item in demo_items[:50]:
                has_gold = len(item.get("gold_spans", [])) > 0
                records.append({
                    "id": f"ragtruth_{item['id']}",
                    "category": "ragtruth_gold",
                    "task": item.get("task", "QA"),
                    "context": item.get("context", ""),
                    "query": item.get("query", ""),
                    "response": item.get("response", ""),
                    "gold_spans": item.get("gold_spans", []),
                    "expected_hallucination": has_gold,
                    "description": f"RAGTruth {item.get('task', 'QA')} passage ({'Hallucinated' if has_gold else 'Grounded'})",
                })

    # ---------------------------------------------------------------------------
    # 2. Contrastive Factual Perturbations (30 items: 15 pairs)
    # ---------------------------------------------------------------------------
    perturb_path = PROJECT_ROOT / "data" / "perturbations.jsonl"
    if perturb_path.exists():
        with open(perturb_path, "r", encoding="utf-8") as f:
            p_items = [json.loads(line) for line in f if line.strip()]

        # Collect 15 pairs (_orig and _pert)
        pair_bases = set()
        for item in p_items:
            base_id = item["id"].replace("_orig", "").replace("_pert", "")
            pair_bases.add(base_id)
            if len(pair_bases) >= 15:
                break

        for item in p_items:
            base_id = item["id"].replace("_orig", "").replace("_pert", "")
            if base_id in pair_bases:
                is_pert = item["id"].endswith("_pert")
                target = item.get("target_span", {})
                gold_spans = [target] if (is_pert and target) else []
                records.append({
                    "id": f"perturb_{item['id']}",
                    "category": "perturbation_pair",
                    "task": item.get("task", "Data2txt"),
                    "context": item.get("context", ""),
                    "query": item.get("query", ""),
                    "response": item.get("response", ""),
                    "gold_spans": gold_spans,
                    "expected_hallucination": is_pert,
                    "description": f"Contrastive Perturbation ({'Altered Conflict' if is_pert else 'Grounded Original'})",
                })

    # ---------------------------------------------------------------------------
    # 3. Open-Domain Factual Queries (20 items)
    # ---------------------------------------------------------------------------
    open_domain_cases = [
        {
            "id": "open_01",
            "query": "Tell me about Google and its origin.",
            "response": "Google was founded on September 4, 1998, by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University in California. The search engine originally began under the name BackRub in 1996. It was later renamed Google, a misspelling of the word googol, which refers to the number 1 followed by one hundred zeros.",
            "expected_hallucination": False,
            "description": "Google origin story (BackRub & Googol definition)",
        },
        {
            "id": "open_02",
            "query": "Tell me about Google's quantum computer Sycamore and its achievement.",
            "response": "In 2019, Google announced that its 53-qubit Sycamore processor achieved quantum supremacy by performing a benchmark calculation in 200 seconds that would take classical supercomputers 10,000 years.",
            "expected_hallucination": False,
            "description": "Sycamore Quantum Supremacy 2019 benchmark",
        },
        {
            "id": "open_03",
            "query": "When was Apollo 11 launched and who walked on the Moon?",
            "response": "Apollo 11 launched on July 16, 1969, carrying commander Neil Armstrong and lunar module pilot Buzz Aldrin, who both walked on the Moon on July 20, 1969.",
            "expected_hallucination": False,
            "description": "Apollo 11 Moon landing historical facts",
        },
        {
            "id": "open_04",
            "query": "Who discovered penicillin and in what year?",
            "response": "Penicillin was discovered in 1928 by Scottish physician and microbiologist Alexander Fleming at St Mary's Hospital in London.",
            "expected_hallucination": False,
            "description": "Alexander Fleming discovery of penicillin in 1928",
        },
        {
            "id": "open_05",
            "query": "What is the speed of light in a vacuum?",
            "response": "The speed of light in a vacuum is exactly 299,792,458 meters per second, commonly rounded to 300,000 kilometers per second.",
            "expected_hallucination": False,
            "description": "Universal constant: speed of light in m/s",
        },
        {
            "id": "open_06",
            "query": "What is the capital of Australia and what is its most populous city?",
            "response": "The capital of Australia is Canberra, while its most populous city is Sydney.",
            "expected_hallucination": False,
            "description": "Canberra vs Sydney geographical facts",
        },
        {
            "id": "open_07",
            "query": "Who developed the theory of General Relativity?",
            "response": "Albert Einstein developed the theory of General Relativity, publishing it in November 1915.",
            "expected_hallucination": False,
            "description": "Albert Einstein General Relativity 1915",
        },
        {
            "id": "open_08",
            "query": "What is the chemical formula for water and table salt?",
            "response": "The chemical formula for water is H2O, and the chemical formula for table salt is NaCl, consisting of sodium and chlorine ions.",
            "expected_hallucination": False,
            "description": "Chemical formulas H2O and NaCl",
        },
        {
            "id": "open_09",
            "query": "When was the World Wide Web invented and by whom?",
            "response": "The World Wide Web was invented in 1989 by British computer scientist Tim Berners-Lee while working at CERN in Switzerland.",
            "expected_hallucination": False,
            "description": "Tim Berners-Lee CERN invention in 1989",
        },
        {
            "id": "open_10",
            "query": "When did World War II end in Europe?",
            "response": "World War II in Europe ended on May 8, 1945, commonly known as Victory in Europe Day (V-E Day), following Nazi Germany's unconditional surrender.",
            "expected_hallucination": False,
            "description": "VE Day May 8 1945 end of WWII in Europe",
        },
        {
            "id": "open_11_hal",
            "query": "When did Apollo 11 land on the moon?",
            "response": "Apollo 11 landed on the lunar surface on October 14, 1982, carrying Neil Armstrong and Buzz Aldrin.",
            "expected_hallucination": True,
            "gold_spans": [{"char_start": 41, "char_end": 57, "text": "October 14, 1982"}],
            "description": "Historical date hallucination (1982 instead of 1969)",
        },
        {
            "id": "open_12_hal",
            "query": "Who was the first president of the United States?",
            "response": "The first president of the United States was Thomas Jefferson, who served from 1789 to 1797.",
            "expected_hallucination": True,
            "gold_spans": [{"char_start": 45, "char_end": 61, "text": "Thomas Jefferson"}],
            "description": "Entity hallucination (Thomas Jefferson instead of George Washington)",
        },
        {
            "id": "open_13_hal",
            "query": "What is the capital of France?",
            "response": "The capital and largest city of France is Berlin, located on the River Seine.",
            "expected_hallucination": True,
            "gold_spans": [{"char_start": 42, "char_end": 48, "text": "Berlin"}],
            "description": "Capital city hallucination (Berlin instead of Paris)",
        },
        {
            "id": "open_14_hal",
            "query": "What is the tallest mountain in the world?",
            "response": "The highest mountain on Earth above sea level is Mount Kilimanjaro, located in the Himalayas with an elevation of 8,848 meters.",
            "expected_hallucination": True,
            "gold_spans": [{"char_start": 49, "char_end": 66, "text": "Mount Kilimanjaro"}],
            "description": "Geographical entity hallucination (Kilimanjaro instead of Everest)",
        },
        {
            "id": "open_15_hal",
            "query": "When was Microsoft founded and who were its founders?",
            "response": "Microsoft was founded on April 4, 1975, by Steve Jobs and Steve Wozniak in Albuquerque, New Mexico.",
            "expected_hallucination": True,
            "gold_spans": [{"char_start": 43, "char_end": 71, "text": "Steve Jobs and Steve Wozniak"}],
            "description": "Corporate founders hallucination (Apple founders instead of Gates & Allen)",
        },
        {
            "id": "open_16",
            "query": "What is DNA and who proposed its double helix structure?",
            "response": "Deoxyribonucleic acid (DNA) is a molecule that carries genetic instructions. In 1953, James Watson and Francis Crick proposed the double helix structure, aided by Rosalind Franklin's X-ray diffraction data.",
            "expected_hallucination": False,
            "description": "DNA double helix structure Watson Crick Franklin 1953",
        },
        {
            "id": "open_17",
            "query": "What is the largest planet in our Solar System?",
            "response": "Jupiter is the largest planet in the Solar System, with a mass more than two and a half times that of all the other planets combined.",
            "expected_hallucination": False,
            "description": "Jupiter largest planet in solar system",
        },
        {
            "id": "open_18",
            "query": "What is the primary currency of Japan?",
            "response": "The primary currency of Japan is the Japanese yen (JPY), which is issued by the Bank of Japan.",
            "expected_hallucination": False,
            "description": "Japanese yen currency",
        },
        {
            "id": "open_19_hal",
            "query": "When did the Titanic sink?",
            "response": "The RMS Titanic sank in the North Atlantic Ocean on the morning of April 15, 1965, after striking an iceberg.",
            "expected_hallucination": True,
            "gold_spans": [{"char_start": 67, "char_end": 81, "text": "April 15, 1965"}],
            "description": "Titanic sinking year hallucination (1965 instead of 1912)",
        },
        {
            "id": "open_20",
            "query": "Who painted the Mona Lisa and in what museum is it displayed?",
            "response": "The Mona Lisa was painted by Italian Renaissance artist Leonardo da Vinci and is permanently displayed at the Louvre Museum in Paris.",
            "expected_hallucination": False,
            "description": "Leonardo da Vinci Louvre Museum Mona Lisa",
        },
    ]

    for item in open_domain_cases:
        records.append({
            "id": f"open_{item['id']}",
            "category": "open_domain",
            "task": "OpenDomainQA",
            "context": "",
            "query": item["query"],
            "response": item["response"],
            "gold_spans": item.get("gold_spans", []),
            "expected_hallucination": item["expected_hallucination"],
            "description": item["description"],
        })

    # ---------------------------------------------------------------------------
    # 4. Scaffolding & Formatting Structural Keys (10 items)
    # ---------------------------------------------------------------------------
    scaffolding_cases = [
        {
            "id": "scaffold_01",
            "context": "The Shard is a 72-storey skyscraper in London, completed in July 2012. It reaches a height of 309.6 meters.",
            "query": "Provide the specifications of The Shard.",
            "response": "**Building**: The Shard\n**Height**: 309.6 meters\n**Completion Year**: 2012\n**Location**: London",
            "expected_hallucination": False,
            "description": "Markdown bold headers (Completion Year, Height, Location)",
        },
        {
            "id": "scaffold_02",
            "context": "Cafe Luna offers free WiFi and outdoor seating. Opening hours are 8:00 AM to 10:00 PM.",
            "query": "List amenities and operating hours for Cafe Luna.",
            "response": "Overview:\n- WiFi: Available\n- Outdoor Seating: Yes\n- Hours: 8:00 AM to 10:00 PM",
            "expected_hallucination": False,
            "description": "Colon key bullets (WiFi, Outdoor Seating, Hours)",
        },
        {
            "id": "scaffold_03_hal",
            "context": "The Shard is a 72-storey skyscraper in London, completed in July 2012. It reaches a height of 309.6 meters.",
            "query": "Provide the specifications of The Shard.",
            "response": "**Building**: The Shard\n**Height**: 540.2 meters\n**Completion Year**: 2012\n**Location**: London",
            "expected_hallucination": True,
            "gold_spans": [{"char_start": 37, "char_end": 49, "text": "540.2 meters"}],
            "description": "Contradictory numeric metric inside structural key value (540.2 meters vs 309.6)",
        },
        {
            "id": "scaffold_04",
            "context": "Inception is a 2010 science fiction film written and directed by Christopher Nolan. It grossed over $836 million worldwide.",
            "query": "Summarize the movie Inception.",
            "response": "Title: Inception\nDirector: Christopher Nolan\nRelease Year: 2010\nBox Office: $836 million",
            "expected_hallucination": False,
            "description": "Film metadata scaffolding (Release Year, Box Office, Director)",
        },
        {
            "id": "scaffold_05_hal",
            "context": "Inception is a 2010 science fiction film written and directed by Christopher Nolan. It grossed over $836 million worldwide.",
            "query": "Summarize the movie Inception.",
            "response": "Title: Inception\nDirector: Christopher Nolan\nRelease Year: 2010\nBox Office: $2.4 billion",
            "expected_hallucination": True,
            "gold_spans": [{"char_start": 68, "char_end": 80, "text": "$2.4 billion"}],
            "description": "Box office monetary hallucination ($2.4 billion vs $836 million)",
        },
        {
            "id": "scaffold_06",
            "context": "Tesla Model 3 has an EPA-estimated range of 341 miles with Long Range all-wheel drive.",
            "query": "What is the range of the Model 3 Long Range?",
            "response": "Model: Tesla Model 3 Long Range\nRange: 341 miles\nDrive: All-Wheel Drive",
            "expected_hallucination": False,
            "description": "Vehicle specification scaffolding (Range, Drive)",
        },
        {
            "id": "scaffold_07_hal",
            "context": "Tesla Model 3 has an EPA-estimated range of 341 miles with Long Range all-wheel drive.",
            "query": "What is the range of the Model 3 Long Range?",
            "response": "Model: Tesla Model 3 Long Range\nRange: 620 miles\nDrive: All-Wheel Drive",
            "expected_hallucination": True,
            "gold_spans": [{"char_start": 39, "char_end": 48, "text": "620 miles"}],
            "description": "Vehicle range hallucination (620 miles vs 341 miles)",
        },
        {
            "id": "scaffold_08",
            "context": "The Eiffel Tower stands 330 meters tall including the radio antenna at the top. Construction finished in 1889.",
            "query": "State the height and finished date of the Eiffel Tower.",
            "response": "Structure: Eiffel Tower\nHeight: 330 meters\nFinished: 1889",
            "expected_hallucination": False,
            "description": "Eiffel Tower metadata keys (Height, Finished)",
        },
        {
            "id": "scaffold_09_hal",
            "context": "The Eiffel Tower stands 330 meters tall including the radio antenna at the top. Construction finished in 1889.",
            "query": "State the height and finished date of the Eiffel Tower.",
            "response": "Structure: Eiffel Tower\nHeight: 330 meters\nFinished: 1945",
            "expected_hallucination": True,
            "gold_spans": [{"char_start": 49, "char_end": 53, "text": "1945"}],
            "description": "Eiffel Tower year hallucination (1945 vs 1889)",
        },
        {
            "id": "scaffold_10",
            "context": "Hotel Bristol features 45 rooms, a rooftop terrace, and on-site parking. Room rates start at €150 per night.",
            "query": "Summarize Hotel Bristol details.",
            "response": "Hotel: Hotel Bristol\nRooms: 45\nPrice: €150 per night\nParking: Available",
            "expected_hallucination": False,
            "description": "Hotel amenity scaffolding (Rooms, Price, Parking)",
        },
    ]

    for item in scaffolding_cases:
        records.append({
            "id": f"scaffold_{item['id']}",
            "category": "scaffolding_keys",
            "task": "StructuredMetadata",
            "context": item.get("context", ""),
            "query": item["query"],
            "response": item["response"],
            "gold_spans": item.get("gold_spans", []),
            "expected_hallucination": item["expected_hallucination"],
            "description": item["description"],
        })

    # Write JSONL
    with open(out_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[Build] Successfully generated {len(records)} benchmark questions -> {out_file}")
    return out_file


if __name__ == "__main__":
    build_benchmark_100()
