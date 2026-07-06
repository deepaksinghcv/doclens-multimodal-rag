"""
app/cli.py — interactive Q&A entry point.

A thin REPL over the online axis: read a question -> answer() -> print grounded answer +
citations. All real work lives in generation/answer.py.

Usage:
    python -m app.cli
""" 

from generation.answer import answer


def main() -> None:
    print("DocLens — ask a question about the indexed manuals.")
    print("Type 'quit' (or empty line) to exit.\n")

    while True:
        question = input("> ").strip()

        if question.lower() in {"", "quit", "exit", "stop","quit()", "exit()", "stop()"} :
            print("Exiting...")
            break

        
        result = answer(question = question)

        print(f'Result: {result["answer"]}\nSource:{result["pages"]}')



if __name__ == "__main__":
    main()
