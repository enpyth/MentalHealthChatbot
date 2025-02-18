import os
from quivr_core import Brain
from dotenv import load_dotenv


def get_brain(question: str):
    data_dir = os.getenv("RAG_DATA_DIR")
    file_paths = [
        os.path.join(data_dir, file)
        for file in os.listdir(data_dir)
        if os.path.isfile(os.path.join(data_dir, file))
    ]

    brain = Brain.from_files(
        name="test_brain",
        file_paths=file_paths,
    )
    res = brain.ask(question)

    return res.answer


if __name__ == "__main__":
    load_dotenv()
    # data_dir = '/Users/zhangsu/workspace/tmp/data'
    res = get_brain("what is my dream about?")
    print(res)
