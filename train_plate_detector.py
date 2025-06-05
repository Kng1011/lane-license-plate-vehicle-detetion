from ultralytics import YOLO
import os
from pathlib import Path
import yaml
import torch

def create_data_yaml():
    """Cria o arquivo data.yaml necessário para o treinamento"""
    data_yaml = {
        'path': 'dataset',  # caminho para o dataset
        'train': 'train/images',  # pasta com imagens de treino
        'val': 'valid/images',    # pasta com imagens de validação
        'test': 'test/images',    # pasta com imagens de teste
        
        'names': {
            0: 'license-plate'  # nome da classe
        }
    }
    
    # Salvar o arquivo data.yaml
    with open('dataset/data.yaml', 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

def train_model():
    """Treina o modelo YOLOv8 para detecção de placas"""
    # Verificar se o arquivo data.yaml existe
    if not Path("dataset/data.yaml").exists():
        create_data_yaml()
    
    # Configurar o modelo
    model = YOLO('yolov8n.pt')  # Começar com um modelo menor para treinamento mais rápido
    
    # Configurar os parâmetros de treinamento
    training_args = {
        'data': 'dataset/data.yaml',  # Arquivo de configuração do dataset
        'epochs': 20,                # Número de épocas
        'imgsz': 420,                # Tamanho das imagens
        'batch': 8,                 # Tamanho do batch
        'name': 'plate_detector',    # Nome do experimento
        'patience': 20,              # Early stopping patience
        'save': True,                # Salvar os melhores modelos
        'device': '0' if torch.cuda.is_available() else 'cpu'  # Usar GPU se disponível
    }
    
    # Iniciar o treinamento
    print("Iniciando treinamento...")
    results = model.train(**training_args)
    
    # Salvar o modelo treinado
    model.save('best.pt')
    print("Modelo treinado salvo como 'best.pt'")

def main():
    """Função principal"""
    print("=== Treinamento do Modelo de Detecção de Placas ===")
    
    # Verificar se o dataset existe
    if not Path("dataset").exists():
        print("Erro: Pasta 'dataset' não encontrada!")
        print("Por favor, certifique-se de que o dataset está na pasta 'dataset'")
        return
    
    # Verificar se o arquivo data.yaml existe
    if not Path("dataset/data.yaml").exists():
        print("Criando arquivo data.yaml...")
        create_data_yaml()
    
    # Treinar o modelo
    train_model()

if __name__ == "__main__":
    main() 