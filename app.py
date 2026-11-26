import json
import os
import re

import gradio as gr
from groq import Groq
from pypdf import PdfReader


MODEL = "openai/gpt-oss-20b"
MAX_NOTE_CHARACTERS = 15000


def get_client():
    """Create the Groq client using an environment variable."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "Groq API key not found. Set the GROQ_API_KEY environment variable."
        )
    return Groq(api_key=api_key)


def read_pdf(file_path):
    if not file_path:
        return "Please choose a PDF file first.", ""

    try:
        reader = PdfReader(file_path)
        pages = []

        for page in reader.pages:
            pages.append(page.extract_text() or "")

        extracted_text = "\n".join(pages).strip()

        if not extracted_text:
            return (
                "No readable text was found. The PDF may be scanned or image-based.",
                ""
            )

        return (
            f"PDF uploaded successfully! Extracted text from {len(reader.pages)} page(s).",
            extracted_text
        )
    except Exception as error:
        return f"Could not read this PDF: {error}", ""


def ask_question(question, notes):
    if not notes:
        return "Please upload your study PDF first."
    if not question or not question.strip():
        return "Please enter a question."

    try:
        response = get_client().chat.completions.create(
            model=MODEL,
            temperature=0.3,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI Study Assistant. Answer using the provided "
                        "study notes. Explain in simple English. If the answer is "
                        "not present in the notes, clearly say so."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"STUDY NOTES:\n{notes[:MAX_NOTE_CHARACTERS]}\n\n"
                        f"STUDENT QUESTION:\n{question}"
                    )
                }
            ]
        )
        return response.choices[0].message.content or "The AI returned an empty answer."
    except Exception as error:
        return f"Could not get an answer. Please check your API key and connection.\n\nDetails: {error}"


def make_summary(notes):
    if not notes:
        return "Please upload your study PDF first."

    try:
        response = get_client().chat.completions.create(
            model=MODEL,
            temperature=0.3,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Summarize study notes in simple English. Use clear headings, "
                        "important points, key definitions, and useful exam points. "
                        "Do not add facts that are not supported by the notes."
                    )
                },
                {
                    "role": "user",
                    "content": notes[:MAX_NOTE_CHARACTERS]
                }
            ]
        )
        return response.choices[0].message.content or "The AI returned an empty summary."
    except Exception as error:
        return f"Could not generate the summary. Please check your API key and connection.\n\nDetails: {error}"


def make_mcqs(notes):
    """Generate and validate exactly five MCQs in JSON format."""
    if not notes:
        raise ValueError("Please upload your study PDF first.")

    response = get_client().chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an exam question generator. Return valid JSON only, "
                    "with no Markdown or text outside the JSON object."
                )
            },
            {
                "role": "user",
                "content": f"""
Create exactly 5 multiple-choice questions using only the study notes below.

Return this JSON structure:
{{
  "questions": [
    {{
      "question": "Question text",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "answer": "The exact text of the correct option",
      "explanation": "A short explanation"
    }}
  ]
}}

Requirements:
- Include exactly 5 questions.
- Each question must have exactly 4 options.
- The answer must exactly match one of the four options.
- Keep questions clear and suitable for student exam practice.

STUDY NOTES:
{notes[:MAX_NOTE_CHARACTERS]}
"""
            }
        ]
    )

    content = response.choices[0].message.content or ""
    content = content.strip()

    # Also handle a response wrapped in Markdown code fences.
    content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
    content = re.sub(r"\s*```$", "", content)

    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("The AI did not return valid quiz JSON. Please generate the quiz again.")

    try:
        data = json.loads(content[start:end + 1])
    except json.JSONDecodeError:
        raise ValueError("The quiz response could not be read. Please generate the quiz again.")

    questions = data.get("questions")
    if not isinstance(questions, list) or len(questions) != 5:
        raise ValueError("The AI did not generate exactly 5 questions. Please try again.")

    for index, item in enumerate(questions, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Question {index} has an invalid format.")

        question_text = item.get("question")
        options = item.get("options")
        answer = item.get("answer")
        explanation = item.get("explanation", "No explanation provided.")

        if not isinstance(question_text, str) or not question_text.strip():
            raise ValueError(f"Question {index} is missing its question text.")
        if not isinstance(options, list) or len(options) != 4:
            raise ValueError(f"Question {index} must contain exactly four options.")
        if not all(isinstance(option, str) and option.strip() for option in options):
            raise ValueError(f"Question {index} contains an invalid option.")
        if answer not in options:
            raise ValueError(f"Question {index} has an answer that does not match its options.")

        item["question"] = question_text.strip()
        item["options"] = options
        item["answer"] = answer
        item["explanation"] = str(explanation)

    return questions


def generate_quiz(notes):
    empty_updates = [gr.update(choices=[], value=None) for _ in range(5)]

    if not notes:
        return "Please upload your study PDF first.", *empty_updates, None

    try:
        questions = make_mcqs(notes)
        question_text = "\n\n".join(
            f"**Question {index}:** {item['question']}"
            for index, item in enumerate(questions, start=1)
        )
        radio_updates = [
            gr.update(choices=item["options"], value=None)
            for item in questions
        ]
        return question_text, *radio_updates, questions
    except Exception as error:
        return f"Quiz generation failed: {error}", *empty_updates, None


def check_answers(questions, answer1, answer2, answer3, answer4, answer5):
    if not questions:
        return "Please generate the quiz first."

    selected_answers = [answer1, answer2, answer3, answer4, answer5]
    score = 0
    feedback = []

    for index, (item, selected) in enumerate(
        zip(questions, selected_answers), start=1
    ):
        correct = item["answer"]

        if selected == correct:
            score += 1
            status = "✅ Correct!"
        elif selected is None:
            status = "⚪ Not answered."
        else:
            status = "❌ Incorrect."

        feedback.append(
            f"**Question {index}: {status}**\n\n"
            f"Your answer: {selected if selected is not None else 'Not answered'}\n\n"
            f"Correct answer: {correct}\n\n"
            f"Explanation: {item['explanation']}"
        )

    return f"# Your Score: {score}/{len(questions)}\n\n" + "\n\n---\n\n".join(feedback)


custom_css = """
.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}
"""

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue",
        neutral_hue="slate"
    ),
    css=custom_css,
    title="AI Study Assistant"
) as app:

    notes_state = gr.State("")
    quiz_state = gr.State(None)

    gr.Markdown(
        """
        # 🎓 AI Study Assistant
        ### Your personal AI-powered learning companion

        Upload your notes, ask questions, create summaries and test your knowledge.
        """
    )

    with gr.Tabs():

        with gr.Tab("📄 Upload PDF"):
            gr.Markdown("### Upload your study material")
            pdf_file = gr.File(
                label="Choose your PDF file",
                file_types=[".pdf"],
                type="filepath"
            )
            upload_button = gr.Button("Upload and Extract Text", variant="primary")
            upload_status = gr.Textbox(label="Upload Status", interactive=False)

            upload_button.click(
                read_pdf,
                inputs=pdf_file,
                outputs=[upload_status, notes_state]
            )

        with gr.Tab("💬 Ask AI"):
            gr.Markdown("### Ask questions from your PDF")
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="Ask anything about your notes...",
                lines=3
            )
            ask_button = gr.Button("Get Answer", variant="primary")
            answer_output = gr.Markdown()

            ask_button.click(
                ask_question,
                inputs=[question_input, notes_state],
                outputs=answer_output
            )

        with gr.Tab("📝 Summarize"):
            gr.Markdown("### Generate study summaries")
            summary_button = gr.Button("Generate Summary", variant="primary")
            summary_output = gr.Markdown()

            summary_button.click(
                make_summary,
                inputs=notes_state,
                outputs=summary_output
            )

        with gr.Tab("🎯 Interactive Quiz"):
            gr.Markdown("### Test your knowledge")
            gr.Markdown("Generate five questions, select your answers and submit.")

            generate_button = gr.Button("Generate Quiz", variant="primary")
            quiz_questions = gr.Markdown()

            answer1 = gr.Radio(label="Question 1", choices=[])
            answer2 = gr.Radio(label="Question 2", choices=[])
            answer3 = gr.Radio(label="Question 3", choices=[])
            answer4 = gr.Radio(label="Question 4", choices=[])
            answer5 = gr.Radio(label="Question 5", choices=[])

            generate_button.click(
                generate_quiz,
                inputs=notes_state,
                outputs=[
                    quiz_questions,
                    answer1,
                    answer2,
                    answer3,
                    answer4,
                    answer5,
                    quiz_state
                ]
            )

            submit_button = gr.Button("Submit Answers", variant="primary")
            quiz_result = gr.Markdown()

            submit_button.click(
                check_answers,
                inputs=[
                    quiz_state,
                    answer1,
                    answer2,
                    answer3,
                    answer4,
                    answer5
                ],
                outputs=quiz_result
            )

    gr.Markdown(
        """
        ---
        **Built with:** Python | Groq API | Gradio | PyPDF
        """
    )


if __name__ == "__main__":
    app.launch()
