# GPTchatApi.py
from flask import Flask, request, jsonify, render_template
from openai import OpenAI
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configurar la carpeta de templates al mismo directorio de este archivo
app = Flask(__name__, template_folder=os.path.dirname(os.path.abspath(__file__)))

# Obtener la API key de OpenAI
api_key = os.getenv('OPENAI_API_KEY')

# Verificar que la API key esté configurada
if not api_key:
    raise ValueError("OPENAI_API_KEY no está configurada. Por favor, crea un archivo .env con tu clave API.")

# Configurar el cliente de OpenAI con la API key
client = OpenAI(api_key=api_key)

# Contexto del asistente especializado en código
SYSTEM_PROMPT = """
Eres un asistente experto en análisis, corrección y creación de código.
Dominas Python, JavaScript, HTML, Java, Docker y otros lenguajes de programación.
Tu objetivo es proporcionar respuestas técnicas claras, concisas y precisas.
Cuando presentes código:
- Usa siempre la sintaxis correcta y formato limpio.
- Incluye comentarios breves y útiles cuando sean necesarios.
- Asegúrate de que el código sea funcional y, si corresponde, optimizado.
Cuando analices código proporcionado por el usuario:
- Explica los errores o mejoras de manera detallada y práctica.
- Propón soluciones con ejemplos de código correctos.
Responde siempre en el mismo idioma en el que escribe el usuario.
"""


@app.route('/')
def index():
    return render_template('GPTchatVisual.html')


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        conversation_history = data.get('history', [])

        # Preparar mensajes para la API
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Agregar historial de conversación
        for msg in conversation_history:
            messages.append({"role": "user", "content": msg['user']})
            if msg['assistant']:  # Solo agregar si hay respuesta
                messages.append({"role": "assistant", "content": msg['assistant']})

        # Agregar el mensaje actual
        messages.append({"role": "user", "content": user_message})

        # Llamar a la API de ChatGPT (nueva sintaxis para openai>=1.0.0)
        response = client.chat.completions.create(
            model="gpt-5-mini-2025-08-07",
            messages=messages,
            temperature=0.0,
            max_completion_tokens=4000
        )

        # Obtener la respuesta (nueva sintaxis)
        assistant_reply = response.choices[0].message.content

        return jsonify({
            'success': True,
            'response': assistant_reply
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/health')
def health_check():
    """Endpoint para verificar que el servidor está funcionando"""
    return jsonify({'status': 'ok', 'api_key_configured': bool(api_key)})


if __name__ == '__main__':
    print("Iniciando servidor Flask...")
    print(f"API key configurada: {'Sí' if api_key else 'No'}")
    if api_key:
        print("✓ OPENAI_API_KEY encontrada correctamente")
    else:
        print("✗ OPENAI_API_KEY no encontrada. Asegúrate de tener un archivo .env")
    app.run(debug=True, port=5000)