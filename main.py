import discord
from discord.ext import commands
import os

# extras para classificação com Keras
import tensorflow as tf
from PIL import Image
import numpy as np
from dotenv import load_dotenv

load_dotenv()

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='$', intents=intents)

@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')

def get_model_path():
    if os.path.exists("./keras_model.h5"):
        return "./keras_model.h5"
    if os.path.exists("./model.savedmodel"):
        return "./model.savedmodel"
    return "./keras_model.h5"


@bot.event
async def on_message(message):
    if message.author.bot:
        return

    if message.content.startswith('$'):
        await bot.process_commands(message)
        return

    if message.attachments:
        for attachment in message.attachments:
            if not attachment.content_type or not attachment.content_type.startswith("image/"):
                continue

            save_path = f"./{attachment.filename}"
            await attachment.save(save_path)
            result = get_class(
                model_path=get_model_path(),
                labels_path="labels.txt",
                image_path=save_path,
            )
            await message.channel.send(f"{message.author.mention}, resultado: {result}")
            return

    await message.channel.send(
        f"Olá, {message.author.mention}! Eu recebi sua mensagem: \"{message.content}\". "
        "Envie uma imagem para eu classificar ou use `$hello` para testar os comandos."
    )

@bot.command()
async def hello(ctx):
    await ctx.send(f'Hi! I am a bot {bot.user}!')

@bot.command()
async def heh(ctx, count_heh = 5):
    await ctx.send("he" * count_heh)


def _load_labels(labels_path):
    try:
        with open(labels_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception:
        return None


def get_class(model_path, labels_path, image_path):
    """Carrega um modelo Keras, pré-processa a imagem e retorna a predição.

    Retorna string com rótulo e confiança quando aplicável.
    """
    # Carrega modelo (suporta .h5 e SavedModel)
    model = tf.keras.models.load_model(model_path)

    # Tenta carregar rótulos
    labels = _load_labels(labels_path)

    # Carrega e pré-processa imagem
    img = Image.open(image_path).convert('RGB')

    # Tenta inferir tamanho de entrada do modelo
    input_shape = model.input_shape
    try:
        if isinstance(input_shape, (list, tuple)):
            if len(input_shape) == 4:
                _, h, w, _ = input_shape
            elif len(input_shape) == 3:
                h, w, _ = input_shape
            else:
                h, w = 224, 224
        else:
            h, w = 224, 224
        if h is None or w is None:
            h, w = 224, 224
    except Exception:
        h, w = 224, 224

    img = img.resize((int(w), int(h)))
    arr = np.array(img).astype('float32') / 255.0
    if arr.ndim == 3:
        arr = np.expand_dims(arr, 0)

    preds = model.predict(arr)

    # Multi-classe (softmax)
    if preds.ndim == 2 and preds.shape[1] > 1:
        probs = preds[0]
        idx = int(np.argmax(probs))
        score = float(probs[idx])
        label = labels[idx] if labels and idx < len(labels) else str(idx)
        return f"{label} ({score:.4f})"

    # Regressão / vetor de outputs
    flat = preds.flatten().tolist()
    if labels and len(labels) == len(flat):
        pairs = [f"{labels[i]}: {flat[i]:.4f}" for i in range(len(flat))]
        return "; ".join(pairs)

    return str(flat)

@bot.command()
async def check(ctx):
    if ctx.message.attachments:
        for attachment in ctx.message.attachments:
            file_name = attachment.filename
            file_url = attachment.url
            save_path = f"./{file_name}"
            await attachment.save(save_path)

            result = get_class(model_path=get_model_path(), labels_path="labels.txt", image_path=save_path)
            await ctx.send(f"{result}\nSaved file: {file_name}\nSource URL: {file_url}")
    else:
        await ctx.send("Envie uma imagem com o comando `$check` para eu classificar.")

# Load bot token from environment for safety. Set DISCORD_BOT_TOKEN in your environment.
if __name__ == '__main__':
    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token or token == "":
        raise RuntimeError(
            "DISCORD_BOT_TOKEN environment variable not set or is still a placeholder. "
            "Create a .env file with DISCORD_BOT_TOKEN=your_token or set the environment variable."
        )

    bot.run(token)
