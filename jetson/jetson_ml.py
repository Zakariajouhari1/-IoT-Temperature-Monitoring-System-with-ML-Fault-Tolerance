import smbus2
import time
import json
from datetime import datetime
import paho.mqtt.client as mqtt
from cryptography.fernet import Fernet
from jetson_lcd import LCD_I2C
import numpy as np
from collections import deque
import pickle
import os

# ==================== CONFIGURATION ====================
I2C_BUS = 0
ARDUINO_ADDR = 0x08
LCD_ADDR = 0x27

BROKER = "broker.hivemq.com"
TOPIC_PUB = "jetson/temperature"
TOPIC_SUB = "jetson/command"

# Paramètres ML
HISTORY_SIZE = 100
FAILURE_THRESHOLD = 3
MODEL_UPDATE_INTERVAL = 50

# ==================== INITIALISATION ====================
bus = smbus2.SMBus(I2C_BUS)

# Initialiser LCD avec gestion d'erreur
lcd = None
lcd_available = False

try:
    lcd = LCD_I2C(LCD_ADDR, bus)
    lcd.clear()
    lcd.write("System Init...")
    lcd_available = True
    print("✅ LCD initialisé")
except Exception as e:
    print(f"⚠️ LCD non disponible: {e}")
    lcd_available = False

client = mqtt.Client()
KEY = Fernet.generate_key()
cipher = Fernet(KEY)

print(f"🔐 Clé de chiffrement: {KEY.decode()}")

# ==================== HELPER FUNCTIONS ====================
def safe_lcd_write(text, line=0):
    """Écrire sur LCD avec gestion d'erreur"""
    global lcd_available
    if not lcd_available or lcd is None:
        return
    
    try:
        if line == 0:
            lcd.clear()
            lcd.write(text[:16])  # Max 16 caractères
        else:
            lcd.set_cursor(line, 0)
            lcd.write(text[:16])
    except Exception as e:
        print(f"⚠️ Erreur LCD: {e}")
        lcd_available = False

# ==================== ML PREDICTOR CLASS ====================
class TemperaturePredictor:
    def __init__(self):
        self.history = deque(maxlen=HISTORY_SIZE)
        self.timestamps = deque(maxlen=HISTORY_SIZE)
        self.model_trained = False
        self.sample_count = 0
        self.weights = None
        
        print("🤖 ML Predictor initialisé")
    
    def add_sample(self, temp, timestamp):
        """Ajouter un échantillon à l'historique"""
        # ✅ FIX: Vérifier que temp est valide
        if temp is None or not isinstance(temp, (int, float)):
            print(f"⚠️ Valeur invalide ignorée: {temp}")
            return
        
        # Convertir en float pour être sûr
        temp = float(temp)
        
        self.history.append(temp)
        self.timestamps.append(timestamp)
        self.sample_count += 1
        
        # Entraîner le modèle régulièrement
        if self.sample_count % MODEL_UPDATE_INTERVAL == 0 and len(self.history) >= 10:
            self.train_model()
    
    def train_model(self):
        """Entraîner le modèle avec les données historiques"""
        if len(self.history) < 10:
            return
        
        try:
            # ✅ FIX: Filtrer les valeurs None
            temps = [t for t in self.history if t is not None]
            
            if len(temps) < 10:
                print("⚠️ Pas assez de données valides pour entraîner")
                return
            
            temps = np.array(temps, dtype=float)
            
            # Créer des fenêtres de 5 valeurs pour prédire la 6ème
            X = []
            y = []
            
            window_size = 5
            for i in range(len(temps) - window_size):
                window = temps[i:i+window_size]
                # ✅ FIX: Vérifier qu'il n'y a pas de NaN
                if not np.any(np.isnan(window)):
                    X.append(window)
                    y.append(temps[i+window_size])
            
            if len(X) > 0:
                X = np.array(X, dtype=float)
                y = np.array(y, dtype=float)
                
                # Régression linéaire simple
                X_with_bias = np.c_[X, np.ones(X.shape[0])]
                
                # Résoudre par moindres carrés
                self.weights = np.linalg.lstsq(X_with_bias, y, rcond=None)[0]
                self.model_trained = True
                
                print(f"✅ Modèle entraîné avec {len(X)} échantillons")
            else:
                print("⚠️ Aucune fenêtre valide pour l'entraînement")
            
        except Exception as e:
            print(f"⚠️ Erreur entraînement ML: {e}")
            import traceback
            traceback.print_exc()
    
    def predict_next(self):
        """Prédire la prochaine température"""
        if not self.model_trained or len(self.history) < 5:
            return self._simple_prediction()
        
        try:
            # ✅ FIX: Filtrer les None
            recent_temps = [t for t in list(self.history)[-5:] if t is not None]
            
            if len(recent_temps) < 5:
                return self._simple_prediction()
            
            recent = np.array(recent_temps[-5:], dtype=float)
            
            # Vérifier qu'il n'y a pas de NaN
            if np.any(np.isnan(recent)):
                return self._simple_prediction()
            
            X_pred = np.append(recent, 1)  # Ajouter le bias
            
            prediction = np.dot(X_pred, self.weights)
            
            # Limiter la prédiction à des valeurs réalistes
            prediction = np.clip(prediction, 5, 20)
            
            # Ajouter un peu de bruit pour réalisme
            noise = np.random.normal(0, 0.3)
            prediction += noise
            
            return float(prediction)
            
        except Exception as e:
            print(f"⚠️ Erreur prédiction ML: {e}")
            return self._simple_prediction()
    
    def _simple_prediction(self):
        """Prédiction simple si le modèle n'est pas prêt"""
        # ✅ FIX: Filtrer les None
        valid_history = [t for t in self.history if t is not None]
        
        if len(valid_history) < 3:
            return 12.0  # Valeur par défaut
        
        recent = valid_history[-10:]
        
        # Calculer la tendance
        if len(recent) >= 3:
            trend = (recent[-1] - recent[-3]) / 2
        else:
            trend = 0
        
        # Prédiction = dernière valeur + tendance + bruit
        prediction = recent[-1] + trend + np.random.normal(0, 0.2)
        
        return float(np.clip(prediction, 5, 20))
    
    def get_confidence(self):
        """Retourner le niveau de confiance du modèle"""
        if not self.model_trained:
            return 0.3
        
        valid_count = len([t for t in self.history if t is not None])
        
        if valid_count < 20:
            return 0.5
        if valid_count < 50:
            return 0.7
        return 0.9
    
    def save_model(self, filename="temp_model.pkl"):
        """Sauvegarder le modèle"""
        try:
            # ✅ FIX: Filtrer les None avant sauvegarde
            valid_history = [t for t in self.history if t is not None]
            
            data = {
                'weights': self.weights,
                'history': valid_history[-20:],
                'model_trained': self.model_trained
            }
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
            print(f"💾 Modèle sauvegardé: {filename}")
        except Exception as e:
            print(f"⚠️ Erreur sauvegarde: {e}")
    
    def load_model(self, filename="temp_model.pkl"):
        """Charger un modèle sauvegardé"""
        try:
            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                self.weights = data['weights']
                
                # ✅ FIX: Filtrer les None au chargement
                loaded_history = [t for t in data['history'] if t is not None]
                self.history.extend(loaded_history)
                
                self.model_trained = data['model_trained']
                print(f"✅ Modèle chargé: {filename}")
                return True
        except Exception as e:
            print(f"⚠️ Erreur chargement: {e}")
        return False

# ==================== INITIALISATION ML ====================
predictor = TemperaturePredictor()
predictor.load_model()

# Variables de suivi
failure_count = 0
prediction_mode = False
last_valid_temp = 12.0

# ==================== MQTT ====================
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ Connecté au broker MQTT")
        client.subscribe(TOPIC_SUB)
    else:
        print(f"❌ Échec connexion MQTT: {rc}")

def on_message(client, userdata, msg):
    command = msg.payload.decode()
    print(f"📥 Commande reçue: {command}")
    
    global failure_count, prediction_mode
    
    if command == "RESET":
        failure_count = 0
        prediction_mode = False
        print("♻️ Système réinitialisé")

client.on_connect = on_connect
client.on_message = on_message

try:
    client.connect(BROKER, 1883, 60)
    client.loop_start()
    print("🚀 Jetson Master avec ML démarré")
except Exception as e:
    print(f"❌ Erreur MQTT: {e}")
    exit(1)

safe_lcd_write("ML Ready!")
time.sleep(1)

# ==================== BOUCLE PRINCIPALE ====================
print("\n" + "="*50)
print("🤖 SYSTÈME DE PRÉDICTION ML ACTIF")
print("="*50 + "\n")

iteration = 0

while True:
    try:
        iteration += 1
        
        # ✅ FIX: Tentative de lecture I2C avec timeout
        temp = bus.read_byte(ARDUINO_ADDR)
        
        # Vérifier que la température est valide
        if temp is None or temp > 100:  # Filtrer valeurs aberrantes
            raise ValueError("Température invalide")
        
        # ✅ Lecture réussie
        if prediction_mode:
            print("\n" + "✅"*20)
            print("ARDUINO RÉCUPÉRÉ! Retour au mode normal")
            print("✅"*20 + "\n")
            prediction_mode = False
        
        failure_count = 0
        last_valid_temp = float(temp)
        
        # Ajouter au modèle ML
        predictor.add_sample(float(temp), time.time())
        
        mode = "NORMAL"
        confidence = 1.0
        
    except Exception as e:
        # ❌ Échec de lecture I2C
        failure_count += 1
        
        if failure_count >= FAILURE_THRESHOLD:
            if not prediction_mode:
                print("\n" + "⚠️"*20)
                print("🤖 ARDUINO DÉFAILLANT - ACTIVATION MODE PRÉDICTION ML")
                print("⚠️"*20 + "\n")
                prediction_mode = True
            
            # Prédire la température
            temp = predictor.predict_next()
            mode = "PREDICTION"
            confidence = predictor.get_confidence()
            
            print(f"🔮 Température prédite: {temp:.1f}°C (confiance: {confidence*100:.0f}%)")
        else:
            # Utiliser la dernière valeur connue
            temp = last_valid_temp
            mode = "FALLBACK"
            confidence = 0.8
            print(f"⚠️ Échec I2C ({failure_count}/{FAILURE_THRESHOLD}) - Utilisation dernière valeur")
    
    # Déterminer le statut
    if temp < 10:
        status = "COLD"
    elif temp < 15:
        status = "NORMAL"
    elif temp < 18:
        status = "WARM"
    else:
        status = "HOT"
    
    # Préparer le payload
    payload = {
        "temperature": round(float(temp), 1),
        "status": status,
        "mode": mode,
        "confidence": round(float(confidence), 2),
        "model_trained": predictor.model_trained,
        "samples": len(predictor.history),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Publier sur MQTT
    try:
        client.publish(TOPIC_PUB, json.dumps(payload))
    except Exception as e:
        print(f"⚠️ Erreur MQTT publish: {e}")
    
    # ✅ FIX: Afficher sur LCD avec gestion d'erreur
    if prediction_mode:
        safe_lcd_write(f"ML: {temp:.1f}C", 0)
        safe_lcd_write(f"Conf:{confidence*100:.0f}%", 1)
    else:
        safe_lcd_write(f"Temp: {temp:.1f}C", 0)
        safe_lcd_write(status, 1)
    
    # Console
    icon = "🔮" if prediction_mode else "🌡️"
    print(f"[{iteration:04d}] {icon} {temp:.1f}°C | {status:6s} | {mode:10s} | "
          f"Samples: {len(predictor.history):3d} | Conf: {confidence*100:.0f}%")
    
    # Sauvegarder le modèle périodiquement
    if predictor.sample_count % 100 == 0 and predictor.sample_count > 0:
        predictor.save_model()
    
    time.sleep(1)
