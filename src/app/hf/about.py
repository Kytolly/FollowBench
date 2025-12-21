import gradio as gr

def create_about_tab():
    with gr.TabItem("📝 About"):
        gr.Markdown("""
        # Ego2Exo with Motion Benchmark
        
        This benchmark evaluates the capability of video generation models to transform **First-Person (Ego)** views into **Third-Person (Exo)** views, focusing on:
        
        - **Quality**: Aesthetic & Imaging Quality
        - **Consistency**: Appearance & Background Semantic Consistency
        - **Motion**: Smoothness & Dynamic Degree
        - **Structure**: Camera Centering & Viewpoint Validity
        """)