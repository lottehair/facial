import os
import time
import json
import base64
import uuid
from pathlib import Path
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename

import cv2
import numpy as np
import mediapipe as mp

# ------------------ Config básico ------------------
BASE = Path(__file__).parent
UPLOAD_DIR = BASE / "static" / "uploads"
SETTINGS_FILE = BASE / "settings.json"
PROPORCOES_FILE = BASE / "proporcoes.json"
ALLOWED = {"png", "jpg", "jpeg"}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

# ------------------ Carrega arquivos externos ------------------
# settings.json (opcional, mas usado para cortes/recomendações)
if SETTINGS_FILE.exists():
    with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
        SETTINGS = json.load(f)
else:
    SETTINGS = {"feminino": {}, "masculino": {}}

# proporcoes.json (obrigatório para classificação)
if not PROPORCOES_FILE.exists():
    raise FileNotFoundError(f"Arquivo de proporções não encontrado: {PROPORCOES_FILE}")

with open(PROPORCOES_FILE, "r", encoding="utf-8") as f:
    PROPORCOES_FORMATOS = json.load(f).get("formatos", {})

# ------------------ Configuração de pesos (importância relativa) ------------------
# Aumente ou diminua pesos conforme preferir. Soma dos pesos não precisa ser 1.
DEFAULT_WEIGHTS = {
    # chaves mais importantes recebem maior peso
    "F_Z": 1.5,
    "Z_M": 1.5,
    "F_M": 1.3,
    "comprimento_rel": 1.5,
    "F_H_total": 1.0,
    "Z_H_total": 1.0,
    "M_H_total": 1.0,
    "WHR": 1.0,
    "golden_ratio_face": 1.2,
    # outras chaves recebem peso menor
    "Z_H": 0.8,
    "H_Z": 0.8,
    "H_F": 0.8,
    "H_M": 0.8,
    "T1_T3": 0.7,
    "T2_T3": 0.7,
    "T1_T2": 0.7,
    "A_mand": 0.5,
    "A_zig": 0.5,
    "simetria_horizontal": 0.6,
    "simetria_vertical": 0.6
}

# Função para obter peso de uma chave (usa default caso não exista)
def get_weight(k):
    return DEFAULT_WEIGHTS.get(k, 0.5)

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# ------------------ Helpers ------------------
def allowed_filename(fname):
    return "." in fname and fname.rsplit(".", 1)[1].lower() in ALLOWED

def save_file_storage(fs):
    filename = secure_filename(fs.filename)
    stamp = str(int(time.time()))
    out = UPLOAD_DIR / f"{stamp}__{filename}"
    fs.save(str(out))
    return out

def save_dataurl(dataurl):
    header, encoded = dataurl.split(",", 1)
    ext = "png" if "png" in header else "jpg"
    b = base64.b64decode(encoded)
    fname = f"{int(time.time())}_{uuid.uuid4().hex}.{ext}"
    path = UPLOAD_DIR / fname
    with open(path, "wb") as f:
        f.write(b)
    return path

def safe_div(a, b, fallback=0.0):
    try:
        if b == 0 or b is None:
            return fallback
        return a / b
    except Exception:
        return fallback

# Score contínuo para comparar um valor com um intervalo [low, high]
# Retorna valor entre 0 e 1, sendo 1 exatamente no intervalo, decrescendo linearmente fora.
def proximity_score(value, low, high, margin_factor=0.5):
    """
    - value: valor observado
    - low, high: intervalo desejado
    - margin_factor: quão longe (em termos de intervalo tamanho) consideramos "próximo" para pontuar parcialmente
    """
    # validação básica
    try:
        low = float(low)
        high = float(high)
        value = float(value)
    except Exception:
        return 0.0

    if low <= value <= high:
        return 1.0

    # intervalo (se zero, utiliza uma margem absoluta pequena)
    interval = max(high - low, 1e-6)
    # margem fora do intervalo onde a pontuação decai até 0
    margin = interval * margin_factor

    if value < low:
        dist = low - value
    else:
        dist = value - high

    if dist >= margin:
        return 0.0
    # pontuação linear decrescente
    return max(0.0, 1.0 - (dist / margin))

# Calcula score total ponderado entre um conjunto de proporcoes observadas e um formato de referência
def compute_weighted_score(proporcoes_obs, formato_ref):
    total_weight = 0.0
    weighted_sum = 0.0
    per_key = {}

    for k, interval in formato_ref.items():
        # intervalo esperado deve ser lista/tupla [low, high]
        if not (isinstance(interval, (list, tuple)) and len(interval) == 2):
            continue
        low, high = interval
        weight = get_weight(k)
        total_weight += weight

        val = proporcoes_obs.get(k, None)
        if val is None:
            score = 0.0
        else:
            score = proximity_score(val, low, high, margin_factor=0.75)  # margem mais generosa
        weighted_sum += score * weight
        per_key[k] = {"value": val, "score": round(score, 4), "weight": weight, "expected": [low, high]}

    # evitar divisão por zero
    if total_weight == 0:
        return {"score_norm": 0.0, "raw": 0.0, "per_key": per_key}

    score_norm = weighted_sum / total_weight  # 0..1
    return {"score_norm": score_norm, "raw": weighted_sum, "per_key": per_key, "total_weight": total_weight}

# ------------------ Lógica principal de análise de imagem ------------------
def analyze_image(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return None, None, None

    img = cv2.convertScaleAbs(img, alpha=1.25, beta=30)
    h, w = img.shape[:2]

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as fm:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = fm.process(rgb)

        if not res.multi_face_landmarks:
            return None, None, None

        lm = res.multi_face_landmarks[0].landmark
        pts = np.array([[p.x * w, p.y * h] for p in lm])

        def p(i): return pts[i]

        # Pontos principais (usando índices do FaceMesh)
        topo = p(10)
        queixo = p(152)
        testa_left = p(127)
        testa_right = p(356)
        bochecha_left = p(234)
        bochecha_right = p(454)
        mandi_left = p(132)
        mandi_right = p(361)

        # Medidas base (pixels)
        comprimento = np.linalg.norm(topo - queixo)
        testa_dist = np.linalg.norm(testa_left - testa_right)
        zig_dist = np.linalg.norm(bochecha_left - bochecha_right)
        mand_dist = np.linalg.norm(mandi_left - mandi_right)

        medidas = {
            "comprimento_px": round(comprimento, 2),
            "testa_px": round(testa_dist, 2),
            "zigomatico_px": round(zig_dist, 2),
            "mandibula_px": round(mand_dist, 2)
        }

        # Calcula proporcoes com proteção contra divisão por zero
        proporcoes = {
            "F_Z": safe_div(testa_dist, zig_dist, 0),
            "Z_M": safe_div(zig_dist, mand_dist, 0),
            "F_M": safe_div(testa_dist, mand_dist, 0),
            "Z_H": safe_div(zig_dist, comprimento, 0),
            "H_Z": safe_div(comprimento, zig_dist, 0),
            "H_F": safe_div(comprimento, testa_dist, 0),
            "H_M": safe_div(comprimento, mand_dist, 0),
            "T1_T3": safe_div(testa_dist, mand_dist, 0),
            "T2_T3": safe_div(zig_dist, mand_dist, 0),
            "T1_T2": safe_div(testa_dist, zig_dist, 0),
            "A_mand": mand_dist,
            "A_zig": zig_dist,
            "F_H_total": safe_div(testa_dist, comprimento, 0),
            "Z_H_total": safe_div(zig_dist, comprimento, 0),
            "M_H_total": safe_div(mand_dist, comprimento, 0),
            "simetria_horizontal": 1.0,  # placeholder (pode ser calculado com landmarks extras)
            "simetria_vertical": 1.0,
            "golden_ratio_face": safe_div(comprimento, (testa_dist + zig_dist + mand_dist) / 3 if (testa_dist + zig_dist + mand_dist) != 0 else 1, 0),
            "WHR": safe_div(zig_dist, comprimento, 0),
            "comprimento_rel": safe_div(comprimento, zig_dist, 0)
        }

        # Comparar com cada formato definido no JSON usando scoring ponderado
        scores_by_format = {}
        for nome, ref in PROPORCOES_FORMATOS.items():
            result = compute_weighted_score(proporcoes, ref)
            scores_by_format[nome] = result

        # Escolher melhor formato (maior score_norm)
        best_format = None
        best_score = -1
        for nome, result in scores_by_format.items():
            if result["score_norm"] > best_score:
                best_score = result["score_norm"]
                best_format = nome

        # calculo de confiança (porcentagem)
        # confidence = best_score normalizado para 0..100
        confidence = round(best_score * 100, 1)

        # Empate ou diferença muito pequena -> desempate por média de distâncias percentuais
        # Ordena por score_norm e verifica segundo colocado
        ordered = sorted(scores_by_format.items(), key=lambda kv: kv[1]["score_norm"], reverse=True)
        if len(ordered) > 1:
            top_name, top_res = ordered[0]
            second_name, second_res = ordered[1]
            diff = top_res["score_norm"] - second_res["score_norm"]
            # se diferença menor que 0.03 (3%) fazemos desempate por soma de (1 - per_key_score) ponderada
            if diff < 0.03:
                def avg_distance(res):
                    # distância média ponderada (quanto menor, melhor)
                    s = 0.0
                    tw = 0.0
                    for k, info in res["per_key"].items():
                        w = info.get("weight", 0.5)
                        sc = info.get("score", 0.0)
                        dist = 1.0 - sc
                        s += dist * w
                        tw += w
                    return (s / tw) if tw != 0 else 1.0
                if avg_distance(top_res) > avg_distance(second_res):
                    # segundo ganha no desempate
                    best_format = second_name
                    best_score = second_res["score_norm"]
                    confidence = round(best_score * 100, 1)

        # Desenhar malha (overlay)
        overlay = img.copy()
        mp_drawing.draw_landmarks(
            overlay,
            res.multi_face_landmarks[0],
            mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
        )

        out_name = f"overlay_{int(time.time())}_{uuid.uuid4().hex}.jpg"
        out_path = UPLOAD_DIR / out_name
        cv2.imwrite(str(out_path), overlay)

        # preparar retorno: simplificar scores por formato para JSON
        scores_summary = {}
        for nome, s in scores_by_format.items():
            scores_summary[nome] = {
                "score_norm": round(s["score_norm"], 4),
                "total_weight": round(s.get("total_weight", 0), 2)
            }

        return best_format, medidas, str(out_path), scores_summary, round(confidence, 1)

# ------------------ Rotas Flask ------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analisar", methods=["POST"])
def analisar():
    genero = request.form.get("genero", "nao_informar")

    uploaded_path = None
    if 'file' in request.files and request.files['file'].filename != "":
        f = request.files['file']
        if not allowed_filename(f.filename):
            return jsonify({"erro": "Formato de arquivo não permitido."}), 400
        uploaded_path = save_file_storage(f)

    elif request.form.get("camera"):
        try:
            uploaded_path = save_dataurl(request.form.get("camera"))
        except Exception:
            return jsonify({"erro": "Imagem da câmera inválida."}), 400
    else:
        return jsonify({"erro": "Nenhuma imagem enviada."}), 400

    try:
        formato, medidas, overlay_path, scores_summary, confidence = analyze_image(uploaded_path)
    except Exception as e:
        # Em caso de erro interno, devolve mensagem JSON clara (ajuda no debug front-end)
        return jsonify({"erro": "Erro interno ao processar a imagem.", "detail": str(e)}), 500

    if not formato:
        return jsonify({"erro": "Rosto não detectado. Centralize e melhore a iluminação."}), 400

    cortes_resp = {}
    if genero == "nao_informar":
        for g in ["feminino", "masculino"]:
            cortes_resp[g] = SETTINGS.get(g, {}).get(formato, [])
    else:
        cortes_resp[genero] = SETTINGS.get(genero, {}).get(formato, [])

    overlay_url = url_for(
        'static',
        filename=str(Path(overlay_path).relative_to(BASE / 'static')).replace("\\", "/")
    )

    return jsonify({
        "formato": formato,
        "medidas": medidas,
        "overlay_url": overlay_url,
        "cortes": cortes_resp,
        "scores": scores_summary,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=True)
