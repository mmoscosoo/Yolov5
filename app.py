import cv2
import streamlit as st
import numpy as np
import pandas as pd
import torch
import os
import sys

st.set_page_config(
    page_title="Detección de Objetos en Tiempo Real - Personalizada",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def load_yolov5_model(model_path='yolov5s.pt'):
    try:
        import yolov5
        try:
            model = yolov5.load(model_path, weights_only=False)
            return model
        except TypeError:
            try:
                model = yolov5.load(model_path)
                return model
            except Exception as e:
                st.warning(f"Intentando un método alternativo de carga...")
                current_dir = os.path.dirname(os.path.abspath(__file__))
                if current_dir not in sys.path:
                    sys.path.append(current_dir)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        st.info("""
        Recomendaciones:
        1. Instalar una versión compatible de PyTorch y YOLOv5:
           ```
           pip install torch==1.12.0 torchvision==0.13.0
           pip install yolov5==7.0.9
           ```
        2. Asegúrate de tener el archivo del modelo en la ubicación correcta
        3. Si el problema persiste, intenta descargar el modelo directamente de torch hub
        """)
        return None

st.title("🔍 Detección de Objetos en Tiempo Real - ¡Intenta con tus imágenes!")
st.markdown("""
¡Usa nuestra herramienta avanzada para detectar objetos en tiempo real! Este sistema está basado en YOLOv5 y ofrece resultados rápidos y precisos. Captura una imagen con tu cámara y ajusta los parámetros para obtener mejores resultados.
""")

with st.spinner("Cargando modelo YOLOv5..."):
    model = load_yolov5_model()

if model:
    st.sidebar.title("🔧 Configuración")
    
    with st.sidebar:
        st.subheader('🔍 Ajustes de Detección', anchor="detectar")
        model.conf = st.slider('Confianza mínima', 0.0, 1.0, 0.25, 0.01, key="confianza", help="Ajusta el umbral de confianza para la detección de objetos.")
        model.iou = st.slider('Umbral IoU', 0.0, 1.0, 0.45, 0.01, key="iou", help="Ajusta el umbral de intersección sobre unión.")
        st.caption(f"Confianza: {model.conf:.2f} | IoU: {model.iou:.2f}")
        
        st.subheader('⚙️ Opciones avanzadas', anchor="avanzadas")
        try:
            model.agnostic = st.checkbox('NMS class-agnostic', False, key="agnostic", help="Activa si quieres usar detección agnóstica de clases.")
            model.multi_label = st.checkbox('Múltiples etiquetas por caja', False, key="multi_label", help="Permite múltiples etiquetas por caja.")
            model.max_det = st.number_input('Detecciones máximas', 10, 2000, 1000, 10, key="max_det", help="Ajusta el número máximo de detecciones.")
        except:
            st.warning("Algunas opciones avanzadas no están disponibles con esta configuración")
    
    main_container = st.container()
    
    with main_container:
        picture = st.camera_input("📸 Captura una imagen", key="camera")
        
        if picture:
            bytes_data = picture.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            with st.spinner("Detectando objetos..."):
                try:
                    results = model(cv2_img)
                except Exception as e:
                    st.error(f"Error durante la detección: {str(e)}")
                    st.stop()
            
            try:
                predictions = results.pred[0]
                boxes = predictions[:, :4]
                scores = predictions[:, 4]
                categories = predictions[:, 5]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🖼 Imagen con detecciones", anchor="imagen")
                    results.render()
                    st.image(cv2_img, channels='BGR', use_column_width=True)
                
                with col2:
                    st.subheader("🔎 Objetos detectados", anchor="detalles")
                    label_names = model.names
                    category_count = {}
                    for category in categories:
                        category_idx = int(category.item()) if hasattr(category, 'item') else int(category)
                        if category_idx in category_count:
                            category_count[category_idx] += 1
                        else:
                            category_count[category_idx] = 1
                    
                    data = []
                    for category, count in category_count.items():
                        label = label_names[category]
                        confidence = scores[categories == category].mean().item() if len(scores) > 0 else 0
                        data.append({
                            "Categoría": label,
                            "Cantidad": count,
                            "Confianza promedio": f"{confidence:.2f}"
                        })
                    
                    if data:
                        df = pd.DataFrame(data)
                        st.dataframe(df, use_container_width=True)
                        
                        st.bar_chart(df.set_index('Categoría')['Cantidad'], color="pink")
                    else:
                        st.info("No se detectaron objetos con los parámetros actuales.")
                        st.caption("Prueba a reducir el umbral de confianza en la barra lateral.")
            except Exception as e:
                st.error(f"Error al procesar los resultados: {str(e)}")
                st.stop()
else:
    st.error("No se pudo cargar el modelo. Por favor verifica las dependencias e inténtalo nuevamente.")
    st.stop()

st.markdown("---")
st.caption("""
**Acerca de la aplicación**: Esta herramienta te permite detectar objetos en tiempo real usando YOLOv5. Ajusta los parámetros para optimizar los resultados. Desarrollada con Streamlit y PyTorch.
""")

