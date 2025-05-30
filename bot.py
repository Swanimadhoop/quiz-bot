# ===================== IMPORTS & SETUP =====================
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
from telegram.constants import ParseMode
import os
import fitz  # PyMuPDF
import nltk
import spacy
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet
from transformers import pipeline, T5Tokenizer
import nest_asyncio
import random

# Initialize the question generation pipeline
qg_pipeline = pipeline(
    "text2text-generation",
    model="valhalla/t5-small-qg-hl",
    tokenizer=T5Tokenizer.from_pretrained("valhalla/t5-small-qg-hl", use_fast=False)
)

nest_asyncio.apply()  # Needed for Colab
BOT_TOKEN = os.getenv("BOT_TOKEN")



nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('omw-1.4')

spacy_model = spacy.load("en_core_web_sm")

# ===================== GLOBAL STATES =====================
user_modes = {}  # user_id -> "qna" / "quiz"
user_data = {}   # user_id -> {"text": ..., "qna_index": ..., "quiz_index": ..., "mcqs": [...]}


# ===================== COMMAND: /start =====================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_modes.pop(user_id, None)
    user_data[user_id] = {"text": "", "qna_index": 0, "quiz_index": 0}
    await update.message.reply_text(
        "👋 Hello! Please send me the text or PDF file first.\n"
        "After that, choose a mode:\n"
        "/qna - Generate questions with answers\n"
        "/quiz - Generate multiple choice questions"
    )

# ===================== MODE SELECTION =====================
async def qna_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    data = user_data.get(user_id)
    if not data or not data["text"]:
        await update.message.reply_text("❌ Please send the text or PDF first.")
        return

    user_modes[user_id] = "qna"
    await update.message.reply_text("🧠 QnA mode activated. Generating questions...")
    await process_qna(update, data["text"])



async def quiz_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    data = user_data.get(user_id)
    if not data or not data["text"]:
        await update.message.reply_text("❌ Please send the text or PDF first.")
        return

    user_modes[user_id] = "quiz"
    await update.message.reply_text("📝 Quiz mode activated. Generating quiz...")
    await process_quiz(update, data["text"])
# ===================== RESET =====================
async def end(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_modes.pop(user_id, None)
    user_data.pop(user_id, None)
    await update.message.reply_text("🛑 Reset. Use /quiz or /qna to start again.")


# ===================== MORE =====================
async def more(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    mode = user_modes.get(user_id)
    data = user_data.get(user_id)

    if not mode or not data or not data["text"]:
        await update.message.reply_text("❌ No previous input found. Use /quiz or /qna and upload content.")
        return

    if mode == "qna":
        await process_qna(update, data["text"], start=data["qna_index"])
    elif mode == "quiz":
        await process_quiz(update, data["text"], start=data["quiz_index"])


# ===================== HANDLE TEXT =====================
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text.strip()

    if not text:
        await update.message.reply_text("❌ No valid text detected.")
        return

    user_data[user_id] = {
        "text": text,
        "qna_index": 0,
        "quiz_index": 0
    }
    await update.message.reply_text("✅ Text received.\nNow choose a mode: /qna or /quiz")


# ===================== HANDLE PDF =====================
async def handle_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    file = await update.message.document.get_file()
    file_path = f"{file.file_id}.pdf"
    await file.download_to_drive(file_path)

    text = extract_text_from_pdf(file_path)
    os.remove(file_path)

    if not text.strip():
        await update.message.reply_text("❌ Could not extract text from PDF.")
        return

    user_data[user_id] = {
        "text": text,
        "qna_index": 0,
        "quiz_index": 0
    }
    await update.message.reply_text("✅ PDF received and processed.\nNow choose a mode: /qna or /quiz")

# ===================== QnA GENERATION =====================
async def process_qna(update: Update, text, start=0, batch_size=5):
    user_id = update.effective_user.id
    qna_pairs = generate_qna_pairs(text)

    if start >= len(qna_pairs):
        await update.message.reply_text("✅ All Q&A pairs sent.")
        return

    for q, a in qna_pairs[start:start + batch_size]:
        await update.message.reply_text(f"❓ *{q}*\n✅ *Answer:* {a}", parse_mode='Markdown')

    user_data[user_id]["qna_index"] = start + batch_size


def generate_qna_pairs(text, limit=30):
    sentences = sent_tokenize(text)
    qna_pairs, seen = [], set()
    for sent in sentences:
        if not (30 <= len(sent) <= 180):
            continue
        doc = spacy_model(sent)
        key_phrases = {chunk.text for chunk in doc.noun_chunks} | {ent.text for ent in doc.ents}
        for phrase in key_phrases:
            if len(phrase.split()) > 6 or len(phrase) < 3 or phrase not in sent:
                continue
            highlighted = sent.replace(phrase, f"<hl> {phrase} <hl>", 1)
            input_text = f"generate question: {highlighted}"
            try:
                result = qg_pipeline(input_text, max_length=64, num_beams=5, early_stopping=True)[0]['generated_text'].strip()
                if result.lower().startswith("question:"):
                    result = result[len("question:"):].strip()
                if not result.endswith('?'):
                    result += '?'
                pair = (result, phrase)
                if pair not in seen:
                    qna_pairs.append(pair)
                    seen.add(pair)
                if len(qna_pairs) >= limit:
                    return qna_pairs
            except Exception as e:
                print(f"⚠️ Error generating question: {e}")
                continue
    return qna_pairs


# ===================== QUIZ GENERATION =====================
async def process_quiz(update: Update, text, start=0, batch_size=5):
    user_id = update.effective_user.id
    all_mcqs = generate_mcqs_from_text(text)

    if start >= len(all_mcqs):
        await update.message.reply_text("✅ All quiz questions sent.")
        return

    for mcq in all_mcqs[start:start + batch_size]:
        # Find the index of the correct answer option
        try:
            correct_idx = mcq['options'].index(mcq['answer'])
        except ValueError:
            correct_idx = 0  # fallback if answer not found in options

        # Send a quiz poll instead of a text message
        await update.message.reply_poll(
            question=mcq['question'],
            options=mcq['options'],
            type='quiz',
            correct_option_id=correct_idx,
            is_anonymous=False,
            allows_multiple_answers=False
        )

    user_data[user_id]["quiz_index"] = start + batch_size

def generate_mcqs_from_text(text):
    all_mcqs = []
    for sent in sent_tokenize(text):
        all_mcqs.extend(generate_mcqs_from_sentence(sent))
    return all_mcqs


def generate_mcqs_from_sentence(sentence):
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)
    mcqs = []
    used_answers = set()

    for word, tag in tagged:
        if tag.startswith('NN') and word.isalpha() and word.lower() not in used_answers:
            distractors = get_distractors_wordnet(word)
            options = distractors + [word]
            options = list(set(options))
            if len(options) < 2:
                continue
            question = replace_word_in_sentence(sentence, word)
            used_answers.add(word.lower())
            random.shuffle(options)
            mcqs.append({
                'question': question,
                'options': options,
                'answer': word
            })
    return mcqs


def get_distractors_wordnet(word):
    distractors = set()
    synsets = wordnet.synsets(word, pos=wordnet.NOUN)
    if not synsets:
        return []
    for lemma in synsets[0].lemmas():
        name = lemma.name().replace('_', ' ')
        if name.lower() != word.lower():
            distractors.add(name)
    return list(distractors)[:3]


def replace_word_in_sentence(sentence, target):
    words = word_tokenize(sentence)
    new_words = []
    replaced = False
    for word in words:
        if not replaced and word.lower() == target.lower():
            new_words.append("_____")
            replaced = True
        else:
            new_words.append(word)
    return " ".join(new_words)


# ===================== ESCAPE MARKDOWN =====================
def escape_markdown(text: str) -> str:
    escape_chars = r"\_*[]()~>#+-=|{}.!"
    return ''.join(['\\' + c if c in escape_chars else c for c in text])


# ===================== PDF TEXT EXTRACTION =====================
def extract_text_from_pdf(path):
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)


# ===================== RUN BOT =====================
app = ApplicationBuilder().token(BOT_TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("qna", qna_mode))
app.add_handler(CommandHandler("quiz", quiz_mode))
app.add_handler(CommandHandler("more", more))
app.add_handler(CommandHandler("end", end))
app.add_handler(MessageHandler(filters.Document.PDF, handle_pdf))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

print("✅ Bot is running...")
app.run_polling()

