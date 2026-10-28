"""
LogitGuard-RAG: 320-Question Comprehensive Multi-Task Benchmark Suite Builder
Compiles a balanced, rigorously categorized benchmark of 320 test questions spanning:
  1. RAGTruth Multi-Task QA/Summary/Data2txt (160 items: 80 with gold hallucination spans, 80 clean)
  2. Contrastive Factual Perturbations (100 items: 50 clean orig, 50 perturbed conflicts)
  3. Open-Domain Authoritative Inquiries (40 items: 20 grounded truths, 20 factual hallucinations)
  4. Scaffolding & Formatting Structural Keys (20 items: 10 clean metadata keys, 10 contradictory values)
"""

import json
from pathlib import Path
from typing import List, Dict, Any

from logitguard.config import PROJECT_ROOT


def build_benchmark_300() -> Path:
    out_file = PROJECT_ROOT / "data" / "benchmark_300.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []

    # ---------------------------------------------------------------------------
    # 1. RAGTruth Multi-Task (160 items: 60 QA, 50 Summary, 50 Data2txt)
    # ---------------------------------------------------------------------------
    ragtruth_path = PROJECT_ROOT / "data" / "ragtruth_clean.jsonl"
    if ragtruth_path.exists():
        qa_hal, qa_clean = [], []
        sum_hal, sum_clean = [], []
        d2t_hal, d2t_clean = [], []

        with open(ragtruth_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                task = item.get("task", "QA")
                has_gold = len(item.get("gold_spans", [])) > 0

                if task == "QA":
                    if has_gold and len(qa_hal) < 30:
                        qa_hal.append(item)
                    elif not has_gold and len(qa_clean) < 30:
                        qa_clean.append(item)
                elif task == "Summary":
                    if has_gold and len(sum_hal) < 25:
                        sum_hal.append(item)
                    elif not has_gold and len(sum_clean) < 25:
                        sum_clean.append(item)
                elif task == "Data2txt":
                    if has_gold and len(d2t_hal) < 25:
                        d2t_hal.append(item)
                    elif not has_gold and len(d2t_clean) < 25:
                        d2t_clean.append(item)

                if (len(qa_hal) == 30 and len(qa_clean) == 30 and
                    len(sum_hal) == 25 and len(sum_clean) == 25 and
                    len(d2t_hal) == 25 and len(d2t_clean) == 25):
                    break

        all_rag = qa_hal + qa_clean + sum_hal + sum_clean + d2t_hal + d2t_clean
        for idx, item in enumerate(all_rag):
            has_gold = len(item.get("gold_spans", [])) > 0
            task = item.get("task", "QA")
            records.append({
                "id": f"ragtruth_{task.lower()}_{idx:03d}",
                "category": "ragtruth_gold",
                "task": task,
                "source_model": item.get("source_model", "gpt-4-0613"),
                "context": item.get("context", ""),
                "query": item.get("query", ""),
                "response": item.get("response", ""),
                "gold_spans": item.get("gold_spans", []),
                "expected_hallucination": has_gold,
                "description": f"RAGTruth {task} sample ({'Hallucinated' if has_gold else 'Grounded'})",
            })

    # ---------------------------------------------------------------------------
    # 2. Contrastive Factual Perturbations (100 items: 50 pairs)
    # ---------------------------------------------------------------------------
    perturb_path = PROJECT_ROOT / "data" / "perturbations.jsonl"
    if perturb_path.exists():
        with open(perturb_path, "r", encoding="utf-8") as f:
            p_items = [json.loads(line) for line in f if line.strip()]

        pair_bases = set()
        for item in p_items:
            base_id = item["id"].replace("_orig", "").replace("_pert", "")
            pair_bases.add(base_id)
            if len(pair_bases) >= 50:
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
                    "source_model": "llama-2-7b-chat",
                    "context": item.get("context", ""),
                    "query": item.get("query", ""),
                    "response": item.get("response", ""),
                    "gold_spans": gold_spans,
                    "expected_hallucination": is_pert,
                    "description": f"Contrastive Perturbation ({'Altered Conflict' if is_pert else 'Grounded Original'})",
                })

    # ---------------------------------------------------------------------------
    # 3. Open-Domain Authoritative Inquiries (40 items: 20 grounded + 20 hallucinations)
    # ---------------------------------------------------------------------------
    open_domain_grounded = [
        ("Tell me about Google and its origin.", "Google was founded on September 4, 1998, by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University in California. The search engine originally began under the name BackRub in 1996. It was later renamed Google, a misspelling of the word googol, which refers to the number 1 followed by one hundred zeros.", "Google origin story (BackRub & Googol definition)"),
        ("Tell me about Google's quantum computer Sycamore and its achievement.", "In 2019, Google announced that its 53-qubit Sycamore processor achieved quantum supremacy by performing a benchmark calculation in 200 seconds that would take classical supercomputers 10,000 years.", "Sycamore Quantum Supremacy 2019 benchmark"),
        ("When was Apollo 11 launched and who walked on the Moon?", "Apollo 11 launched on July 16, 1969, carrying commander Neil Armstrong and lunar module pilot Buzz Aldrin, who both walked on the Moon on July 20, 1969.", "Apollo 11 Moon landing historical facts"),
        ("Who discovered penicillin and in what year?", "Penicillin was discovered in 1928 by Scottish physician and microbiologist Alexander Fleming at St Mary's Hospital in London.", "Alexander Fleming discovery of penicillin in 1928"),
        ("What is the speed of light in a vacuum?", "The speed of light in a vacuum is exactly 299,792,458 meters per second, commonly rounded to 300,000 kilometers per second.", "Universal constant: speed of light in m/s"),
        ("What is the capital of Australia and what is its most populous city?", "The capital of Australia is Canberra, while its most populous city is Sydney.", "Canberra vs Sydney geographical facts"),
        ("Who developed the theory of General Relativity?", "Albert Einstein developed the theory of General Relativity, publishing it in November 1915.", "Albert Einstein General Relativity 1915"),
        ("What is the chemical formula for water and table salt?", "The chemical formula for water is H2O, and the chemical formula for table salt is NaCl, consisting of sodium and chlorine ions.", "Chemical formulas H2O and NaCl"),
        ("When was the World Wide Web invented and by whom?", "The World Wide Web was invented in 1989 by British computer scientist Tim Berners-Lee while working at CERN in Switzerland.", "Tim Berners-Lee CERN invention in 1989"),
        ("When did World War II end in Europe?", "World War II in Europe ended on May 8, 1945, commonly known as Victory in Europe Day (V-E Day), following Nazi Germany's unconditional surrender.", "VE Day May 8 1945 end of WWII in Europe"),
        ("What is DNA and who proposed its double helix structure?", "Deoxyribonucleic acid (DNA) is a molecule that carries genetic instructions. In 1953, James Watson and Francis Crick proposed the double helix structure, aided by Rosalind Franklin's X-ray diffraction data.", "DNA double helix structure Watson Crick Franklin 1953"),
        ("What is the largest planet in our Solar System?", "Jupiter is the largest planet in the Solar System, with a mass more than two and a half times that of all the other planets combined.", "Jupiter largest planet in solar system"),
        ("What is the primary currency of Japan?", "The primary currency of Japan is the Japanese yen (JPY), which is issued by the Bank of Japan.", "Japanese yen currency"),
        ("Who painted the Mona Lisa and in what museum is it displayed?", "The Mona Lisa was painted by Italian Renaissance artist Leonardo da Vinci and is permanently displayed at the Louvre Museum in Paris.", "Leonardo da Vinci Louvre Museum Mona Lisa"),
        ("What is the process by which plants make food?", "Photosynthesis is the biochemical process by which green plants convert light energy, carbon dioxide, and water into glucose and oxygen using chlorophyll.", "Photosynthesis chemical biological definition"),
        ("What is the circumference of Earth at the equator?", "The circumference of Earth at the equator is approximately 40,075 kilometers (about 24,901 miles).", "Earth equatorial circumference in km and miles"),
        ("Who invented the telephone?", "Alexander Graham Bell was awarded the first U.S. patent for the telephone in March 1876.", "Alexander Graham Bell telephone patent 1876"),
        ("Who developed the movable-type printing press in Europe?", "Johannes Gutenberg introduced movable-type printing to Europe around 1440, revolutionizing information dissemination.", "Johannes Gutenberg printing press 1440"),
        ("Who created the Periodic Table of elements?", "Russian chemist Dmitri Mendeleev published the first widely recognized Periodic Table in 1869, arranging elements by atomic mass.", "Dmitri Mendeleev periodic table 1869"),
        ("Who improved the steam engine facilitating the Industrial Revolution?", "Scottish engineer James Watt significantly improved the Newcomen steam engine in 1769 by introducing a separate condenser.", "James Watt steam engine condenser 1769"),
    ]

    open_domain_hallucinated = [
        ("When did Apollo 11 land on the moon?", "Apollo 11 landed on the lunar surface on October 14, 1982, carrying Neil Armstrong and Buzz Aldrin.", "October 14, 1982", "Date error: 1982 vs 1969"),
        ("Who was the first president of the United States?", "The first president of the United States was Thomas Jefferson, who served from 1789 to 1797.", "Thomas Jefferson", "Entity swap: Jefferson vs Washington"),
        ("What is the capital of France?", "The capital and largest city of France is Berlin, located on the River Seine.", "Berlin", "Geographical error: Berlin vs Paris"),
        ("What is the tallest mountain in the world?", "The highest mountain on Earth above sea level is Mount Kilimanjaro, located in the Himalayas with an elevation of 8,848 meters.", "Mount Kilimanjaro", "Entity error: Kilimanjaro vs Everest"),
        ("When was Microsoft founded and who were its founders?", "Microsoft was founded on April 4, 1975, by Steve Jobs and Steve Wozniak in Albuquerque, New Mexico.", "Steve Jobs and Steve Wozniak", "Entity swap: Apple founders vs Gates & Allen"),
        ("When did the Titanic sink?", "The RMS Titanic sank in the North Atlantic Ocean on the morning of April 15, 1965, after striking an iceberg.", "April 15, 1965", "Date error: 1965 vs 1912"),
        ("What is the largest planet in our solar system?", "The largest planet in the Solar System is Mars, with a diameter exceeding 140,000 kilometers.", "Mars", "Planetary swap: Mars vs Jupiter"),
        ("Who invented the World Wide Web?", "The World Wide Web was invented in 1989 by Bill Gates while working as an engineer at Microsoft headquarters.", "Bill Gates", "Entity swap: Gates vs Berners-Lee"),
        ("When did World War I end?", "World War I officially concluded with the signing of the armistice on November 11, 1945.", "November 11, 1945", "Year error: 1945 vs 1918"),
        ("Who discovered the law of universal gravitation?", "The law of universal gravitation was formulated by Marie Curie in 1687 in her mathematical treatise.", "Marie Curie", "Entity swap: Curie vs Newton"),
        ("What is the chemical symbol for gold?", "The chemical symbol for gold is Fe, derived from its Latin name ferrum.", "Fe", "Chemical symbol swap: Fe (Iron) vs Au (Gold)"),
        ("Who painted the Sistine Chapel ceiling?", "The ceiling of the Sistine Chapel was painted by Vincent van Gogh between 1508 and 1512.", "Vincent van Gogh", "Artist swap: Van Gogh vs Michelangelo"),
        ("What year was the Python programming language first released?", "Python was created by Guido van Rossum and first released to the public in February 1950.", "February 1950", "Year error: 1950 vs 1991"),
        ("Who discovered penicillin?", "Penicillin was discovered in 1928 by French chemist Louis Pasteur in his laboratory in Paris.", "Louis Pasteur", "Scientist swap: Pasteur vs Fleming"),
        ("What is the capital of Canada?", "The capital city of Canada is Vancouver, located in the province of Ontario.", "Vancouver", "City error: Vancouver vs Ottawa"),
        ("Who wrote the play Hamlet?", "The tragedy Hamlet was written by English author Charles Dickens in the early 17th century.", "Charles Dickens", "Author swap: Dickens vs Shakespeare"),
        ("What is the hardest natural mineral on Earth?", "The hardest known natural mineral on the Mohs scale is Talc, with a hardness rating of 10.", "Talc", "Mineral error: Talc (softest) vs Diamond (hardest)"),
        ("What is the capital of Japan?", "The capital and largest metropolis of Japan is Kyoto, which has served as the capital continuously since 1868.", "Kyoto", "City swap: Kyoto vs Tokyo"),
        ("When did humans first land on Mars?", "NASA's Apollo 18 mission successfully landed human astronauts on Mars on July 20, 1979.", "Apollo 18 mission successfully landed human astronauts on Mars on July 20, 1979", "Counterfactual historical event"),
        ("What is the speed of sound in air?", "The speed of sound in dry air at 20 degrees Celsius is approximately 300,000 kilometers per second.", "300,000 kilometers per second", "Physical constant confusion: speed of sound vs speed of light"),
    ]

    for idx, (q, resp, desc) in enumerate(open_domain_grounded):
        records.append({
            "id": f"open_grounded_{idx+1:02d}",
            "category": "open_domain",
            "task": "OpenDomainQA",
            "context": "",
            "query": q,
            "response": resp,
            "gold_spans": [],
            "expected_hallucination": False,
            "description": desc,
        })

    for idx, (q, resp, err_text, desc) in enumerate(open_domain_hallucinated):
        s_idx = resp.find(err_text)
        gold_spans = []
        if s_idx != -1:
            gold_spans.append({"char_start": s_idx, "char_end": s_idx + len(err_text), "text": err_text})
        records.append({
            "id": f"open_hal_{idx+1:02d}",
            "category": "open_domain",
            "task": "OpenDomainQA",
            "context": "",
            "query": q,
            "response": resp,
            "gold_spans": gold_spans,
            "expected_hallucination": True,
            "description": desc,
        })

    # ---------------------------------------------------------------------------
    # 4. Scaffolding & Formatting Structural Keys (20 items: 10 clean, 10 hallucinated)
    # ---------------------------------------------------------------------------
    scaffolding_clean = [
        ("The Shard is a 72-storey skyscraper in London, completed in July 2012. It reaches a height of 309.6 meters.", "Provide the specifications of The Shard.", "**Building**: The Shard\n**Height**: 309.6 meters\n**Completion Year**: 2012\n**Location**: London", "The Shard building specs"),
        ("Cafe Luna offers free WiFi and outdoor seating. Opening hours are 8:00 AM to 10:00 PM.", "List amenities and operating hours for Cafe Luna.", "Overview:\n- WiFi: Available\n- Outdoor Seating: Yes\n- Hours: 8:00 AM to 10:00 PM", "Cafe Luna amenity list"),
        ("Inception is a 2010 science fiction film written and directed by Christopher Nolan. It grossed over $836 million worldwide.", "Summarize the movie Inception.", "Title: Inception\nDirector: Christopher Nolan\nRelease Year: 2010\nBox Office: $836 million", "Inception film specs"),
        ("Tesla Model 3 has an EPA-estimated range of 341 miles with Long Range all-wheel drive.", "What is the range of the Model 3 Long Range?", "Model: Tesla Model 3 Long Range\nRange: 341 miles\nDrive: All-Wheel Drive", "Tesla Model 3 specs"),
        ("The Eiffel Tower stands 330 meters tall including the radio antenna at the top. Construction finished in 1889.", "State the height and finished date of the Eiffel Tower.", "Structure: Eiffel Tower\nHeight: 330 meters\nFinished: 1889", "Eiffel Tower specs"),
        ("Hotel Bristol features 45 rooms, a rooftop terrace, and on-site parking. Room rates start at €150 per night.", "Summarize Hotel Bristol details.", "Hotel: Hotel Bristol\nRooms: 45\nPrice: €150 per night\nParking: Available", "Hotel Bristol specs"),
        ("Space Needle stands at 184 meters (605 feet) tall and opened in 1962 for the World's Fair in Seattle.", "Summarize Space Needle details.", "Observation Tower: Space Needle\nCity: Seattle\nHeight: 184 meters\nOpened: 1962", "Space Needle specs"),
        ("Saturn V rocket was 110.6 meters (363 feet) tall with a payload capacity of 140,000 kg to Low Earth Orbit.", "Provide Saturn V technical parameters.", "Vehicle: Saturn V\nHeight: 110.6 meters\nLEO Payload: 140,000 kg", "Saturn V rocket specs"),
        ("Burj Khalifa in Dubai reaches 828 meters (2,717 feet) with 163 floors, completed in 2010.", "List Burj Khalifa specifications.", "Structure: Burj Khalifa\nCity: Dubai\nHeight: 828 meters\nFloors: 163\nCompleted: 2010", "Burj Khalifa specs"),
        ("Golden Gate Bridge has a total length of 2,737 meters (8,981 feet) and opened in May 1937 in San Francisco.", "Provide Golden Gate Bridge specifications.", "Bridge: Golden Gate Bridge\nLocation: San Francisco\nTotal Length: 2,737 meters\nOpened: 1937", "Golden Gate Bridge specs"),
    ]

    scaffolding_hallucinated = [
        ("The Shard is a 72-storey skyscraper in London, completed in July 2012. It reaches a height of 309.6 meters.", "Provide the specifications of The Shard.", "**Building**: The Shard\n**Height**: 540.2 meters\n**Completion Year**: 2012\n**Location**: London", "540.2 meters", "Height mutation: 540.2 vs 309.6"),
        ("Inception is a 2010 science fiction film written and directed by Christopher Nolan. It grossed over $836 million worldwide.", "Summarize the movie Inception.", "Title: Inception\nDirector: Christopher Nolan\nRelease Year: 2010\nBox Office: $2.4 billion", "$2.4 billion", "Box office mutation: $2.4B vs $836M"),
        ("Tesla Model 3 has an EPA-estimated range of 341 miles with Long Range all-wheel drive.", "What is the range of the Model 3 Long Range?", "Model: Tesla Model 3 Long Range\nRange: 620 miles\nDrive: All-Wheel Drive", "620 miles", "Range mutation: 620 vs 341"),
        ("The Eiffel Tower stands 330 meters tall including the radio antenna at the top. Construction finished in 1889.", "State the height and finished date of the Eiffel Tower.", "Structure: Eiffel Tower\nHeight: 330 meters\nFinished: 1945", "1945", "Finished year mutation: 1945 vs 1889"),
        ("Hotel Bristol features 45 rooms, a rooftop terrace, and on-site parking. Room rates start at €150 per night.", "Summarize Hotel Bristol details.", "Hotel: Hotel Bristol\nRooms: 45\nPrice: €950 per night\nParking: Available", "€950 per night", "Price mutation: €950 vs €150"),
        ("Space Needle stands at 184 meters (605 feet) tall and opened in 1962 for the World's Fair in Seattle.", "Summarize Space Needle details.", "Observation Tower: Space Needle\nCity: Seattle\nHeight: 450 meters\nOpened: 1962", "450 meters", "Height mutation: 450m vs 184m"),
        ("Saturn V rocket was 110.6 meters (363 feet) tall with a payload capacity of 140,000 kg to Low Earth Orbit.", "Provide Saturn V technical parameters.", "Vehicle: Saturn V\nHeight: 30.5 meters\nLEO Payload: 140,000 kg", "30.5 meters", "Height mutation: 30.5m vs 110.6m"),
        ("Burj Khalifa in Dubai reaches 828 meters (2,717 feet) with 163 floors, completed in 2010.", "List Burj Khalifa specifications.", "Structure: Burj Khalifa\nCity: Dubai\nHeight: 1,420 meters\nFloors: 163\nCompleted: 2010", "1,420 meters", "Height mutation: 1420m vs 828m"),
        ("Golden Gate Bridge has a total length of 2,737 meters (8,981 feet) and opened in May 1937 in San Francisco.", "Provide Golden Gate Bridge specifications.", "Bridge: Golden Gate Bridge\nLocation: San Francisco\nTotal Length: 8,400 meters\nOpened: 1937", "8,400 meters", "Length mutation: 8400m vs 2737m"),
        ("Cafe Luna offers free WiFi and outdoor seating. Opening hours are 8:00 AM to 10:00 PM.", "List amenities and operating hours for Cafe Luna.", "Overview:\n- WiFi: Not Available\n- Outdoor Seating: Yes\n- Hours: 8:00 AM to 10:00 PM", "Not Available", "Amenity negation flip: Not Available vs Available"),
    ]

    for idx, (ctx, q, resp, desc) in enumerate(scaffolding_clean):
        records.append({
            "id": f"scaffold_clean_{idx+1:02d}",
            "category": "scaffolding_keys",
            "task": "StructuredMetadata",
            "context": ctx,
            "query": q,
            "response": resp,
            "gold_spans": [],
            "expected_hallucination": False,
            "description": desc,
        })

    for idx, (ctx, q, resp, err_text, desc) in enumerate(scaffolding_hallucinated):
        s_idx = resp.find(err_text)
        gold_spans = []
        if s_idx != -1:
            gold_spans.append({"char_start": s_idx, "char_end": s_idx + len(err_text), "text": err_text})
        records.append({
            "id": f"scaffold_hal_{idx+1:02d}",
            "category": "scaffolding_keys",
            "task": "StructuredMetadata",
            "context": ctx,
            "query": q,
            "response": resp,
            "gold_spans": gold_spans,
            "expected_hallucination": True,
            "description": desc,
        })

    # Write out benchmark
    with open(out_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[BenchmarkBuilder] Generated {len(records)} benchmark items in {out_file}")
    return out_file


if __name__ == "__main__":
    build_benchmark_300()
