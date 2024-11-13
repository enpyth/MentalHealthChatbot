from nodes import app
from IPython.display import Image
from langchain_core.runnables.graph import MermaidDrawMethod


def display_mermaid():
    print(app.get_graph().draw_mermaid())
    graph_img = Image(
        app.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
        )
    )
    with open("graph_img.png", "wb") as f:
        f.write(graph_img.data)


if __name__ == "__main__":
    display_mermaid()
