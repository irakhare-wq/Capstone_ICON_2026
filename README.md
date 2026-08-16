# Capstone_ICON_2026
A Quality Assurance Framework for AI Retrieval Systems in Oncology Regulatory Guidance

Repository Structure
FDA documents/: the six source FDA guidance documents used by the guidance pipeline.
Evaluation_results.xlsx: the evaluation questions and results used to test the guidance pipeline.
Guidance_Pipeline.py: the FDA guidance RAG pipeline: document ingestion, chunking, retrieval, answer generation, and the eleven-grader quality assurance framework.
Guidance_App.py: the web application for the guidance pipeline, letting a reviewer ask a question, view the generated answer, and inspect the source evidence and quality scores behind it.
Protocols_Pipeline.py: the protocol-extension pipeline: structure-aware parsing, hybrid retrieval and scope routing, adapted for long, deeply nested clinical trial protocols.
Protocols_App.py: the web application for the protocol pipeline.

Tech Stack
Generation model: Azure OpenAI, gpt-5.4-mini
Groundedness judge panel: deepseek-chat and Groq's llama-3.1-8b-instant
Automated relevancy metric: DeepEval (AnswerRelevancyMetric)
