import gradio as gr

def create_citation_tab():
    with gr.TabItem("📚 Citation"):
        gr.Markdown("### BibTeX")
        gr.Code("""
            @article{ego2exo2025,
            title={Ego2Exo In Follow Camera: A Benchmark for Egocentric-to-Exocentric Video Generation},
            author={FYP21NUSCQ},
            journal={arXiv preprint},
            year={2025}
            }
        """, language=None)