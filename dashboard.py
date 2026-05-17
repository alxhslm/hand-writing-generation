import numpy as np
import streamlit as st
import torch
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from training.model import VAE
from training.utils import class_num_to_label, generate_images, label_to_class_num, recenter_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

st.set_page_config("Hand-writing generator", page_icon="writing_hand", layout="wide")


@st.cache_resource()
def load_model(filename: str) -> VAE:
    model = VAE(input_size=28, output_size=10 + 26, num_filters=32, num_latent_var=64).to(device)
    model.load_state_dict(torch.load(filename, map_location=device))
    model.eval()
    return model




st.title(":writing_hand: Hand-writing generator")
st.markdown(
    "This is an ML-powered app for generating synthetic hand-written characters in a given writing style. "
    "You can find out more about how it works [here](https://alxhslm.github.io/projects/hand-writing-generation/)."
)

st.header(":mag: Identifying your writing style")
st.markdown("First draw some characters in the boxes below. These will be used to identify the style of your writing.")
st.markdown(
    "Here are some things to try: \n"
    " - Writing in *italic* \n"
    " - Varying the width of your characters \n"
    " - Changing stroke thickness"
)


stroke_width = st.slider("Stroke thickness: ", min_value=1, max_value=25, value=20)
columns = st.columns(5)
input_images = []
for i, col in enumerate(columns):
    with col:
        image = Image.fromarray(
            st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                stroke_width=stroke_width,
                stroke_color="white",
                background_color="black",
                height=150,
                width=150,
                key=f"input_character_{i}",
            ).image_data
        )
        image = image.convert("L")
        image = recenter_image(image)
        input_images.append(np.expand_dims(np.array(image) / 255, axis=0))


z_means = []
log_z_vars = []
model = load_model("model.pt")
for col, im in zip(columns, input_images):
    x = torch.Tensor([im])
    y_pred, z_mean, log_z_var = model.encode(x)
    y_prob = y_pred.softmax(dim=1)[0, :].detach()
    imax = y_prob.argmax().numpy()
    label = "{} ({:.2f}%)".format(class_num_to_label(imax), y_prob[imax] * 100)
    col.text(label)
    z_means.append(z_mean[0, :].detach())
    log_z_vars.append(log_z_var[0, :].detach())

z_mean = torch.stack(z_means)
log_z_var = torch.stack(log_z_vars)

st.header(":memo: Generating synthetic characters")
st.subheader(":keyboard: Decide what to generate")
mode = st.radio("Generate", options=["Full character set", "Sentence"], horizontal=True)
if mode == "Full character set":
    text = "".join([class_num_to_label(i) for i in range(0, model.output_size)])
elif mode == "Sentence":
    text = st.text_area(
        "Text to generate",
        value="The quick brown fox jumps over the lazy dog",
        help="Alphabetic characters will be converted to upepr case. Other characters will be ignored.",
    )
else:
    text = ""

st.subheader(":printer: Synthesize characters")
with st.expander("Options"):
    randomness = st.slider("Randomness: ", min_value=0.0, value=1.0)
    if st.checkbox("Compute average latent variable"):
        z_mean = z_mean.mean(dim=0).unsqueeze(0)
        log_z_var = log_z_var.mean(dim=0).unsqueeze(0)

images = []
for t in text:
    index = label_to_class_num(t.upper())
    if index:
        images.append(generate_images(model, torch.tensor([index]), z_mean, log_z_var, randomness)[0])
    else:
        images.append(np.zeros((28, 28)))

if images:
    st.image(np.concatenate(images, axis=-1))
