**🔍 Interactive Image Processing App**

This is a **Streamlit-based web app** that allows users to apply various **edge detection, feature detection, and segmentation techniques** on uploaded images.

**🚀 Features**

- **Edge Detection**: Sobel, Prewitt, Canny, and Laplace
- **Feature Detection**: Point Detection & Line Detection (Vertical, Horizontal, ±45°)
- **Segmentation**: Watershed-based image segmentation
- **User Interaction**: Adjustable scale sliders for fine-tuning results
- **Quick Quiz**: Test your knowledge on image processing

**🛠️ Installation**

**1️⃣ Clone the Repository**

sh

CopyEdit

git clone <https://github.com/your-username/your-repo-name.git>

cd your-repo-name

**2️⃣ Install Dependencies**

sh

CopyEdit

pip install -r requirements.txt

**3️⃣ Run the App**

sh

CopyEdit

streamlit run main2.py

**🌍 Deploy on Render**

**1️⃣ Create requirements.txt**

sh

CopyEdit

pip freeze > requirements.txt

**2️⃣ Create render.yaml**

For automatic deployment, create a render.yaml file and add:

yaml

CopyEdit

services:

\- type: web

name: image-processing-app

env: python

buildCommand: "pip install -r requirements.txt"

startCommand: "streamlit run main2.py"

**3️⃣ Deploy on Render**

- **Push your code to GitHub**
- **Go to** [**Render**](https://render.com/)
- **Create a new Web Service**
- **Connect your GitHub repository**
- **Deploy 🚀**

**📌 Example**

Upload an image and explore different processing techniques! 🎨🔬
