"""
Script para generar figuras de H2: Data Augmentation
Ejecutar desde la raíz del proyecto: python generate_h2_figures.py
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd

# Configurar estilo
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 12

# Rutas
OUTPUT_DIR = Path("Report/Figures/H2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Colores
COLOR_SIN_AUG = '#2ecc71'  # Verde
COLOR_CON_AUG = '#e74c3c'  # Rojo

def load_checkpoint_metadata(exp_id):
    """Cargar metadatos de checkpoints"""
    path = Path(f"checkpoints/{exp_id}/checkpoints_metadata.json")
    with open(path, 'r') as f:
        data = json.load(f)
    return data['checkpoints']

def plot_training_curves():
    """Figura 1: Comparación de curvas de entrenamiento"""
    # Cargar datos
    con_aug = load_checkpoint_metadata("exp_10ed8760")  # CON augmentation
    sin_aug = load_checkpoint_metadata("exp_ddde3a41")  # SIN augmentation
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- Subplot 1: Loss ---
    ax1 = axes[0]
    
    # SIN augmentation
    epochs_sin = [cp['epoch'] for cp in sin_aug]
    train_loss_sin = [cp['metrics']['train_loss'] for cp in sin_aug]
    val_loss_sin = [cp['metrics']['val_loss'] for cp in sin_aug]
    
    ax1.plot(epochs_sin, train_loss_sin, '-', color=COLOR_SIN_AUG, label='Train (SIN Aug)', linewidth=2)
    ax1.plot(epochs_sin, val_loss_sin, '--', color=COLOR_SIN_AUG, label='Val (SIN Aug)', linewidth=2)
    
    # CON augmentation
    epochs_con = [cp['epoch'] for cp in con_aug]
    train_loss_con = [cp['metrics']['train_loss'] for cp in con_aug]
    val_loss_con = [cp['metrics']['val_loss'] for cp in con_aug]
    
    ax1.plot(epochs_con, train_loss_con, '-', color=COLOR_CON_AUG, label='Train (CON Aug)', linewidth=2)
    ax1.plot(epochs_con, val_loss_con, '--', color=COLOR_CON_AUG, label='Val (CON Aug)', linewidth=2)
    
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Loss')
    ax1.set_title('Curvas de Pérdida')
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 0.6)
    
    # --- Subplot 2: Accuracy ---
    ax2 = axes[1]
    
    # SIN augmentation
    train_acc_sin = [cp['metrics']['train_acc'] * 100 for cp in sin_aug]
    val_acc_sin = [cp['metrics']['val_acc'] * 100 for cp in sin_aug]
    
    ax2.plot(epochs_sin, train_acc_sin, '-', color=COLOR_SIN_AUG, label='Train (SIN Aug)', linewidth=2)
    ax2.plot(epochs_sin, val_acc_sin, '--', color=COLOR_SIN_AUG, label='Val (SIN Aug)', linewidth=2)
    
    # CON augmentation
    train_acc_con = [cp['metrics']['train_acc'] * 100 for cp in con_aug]
    val_acc_con = [cp['metrics']['val_acc'] * 100 for cp in con_aug]
    
    ax2.plot(epochs_con, train_acc_con, '-', color=COLOR_CON_AUG, label='Train (CON Aug)', linewidth=2)
    ax2.plot(epochs_con, val_acc_con, '--', color=COLOR_CON_AUG, label='Val (CON Aug)', linewidth=2)
    
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Curvas de Accuracy')
    ax2.legend(loc='lower right')
    ax2.set_ylim(60, 102)
    
    # Añadir anotaciones
    ax2.annotate(f'SIN Aug: {val_acc_sin[-1]:.1f}%', 
                 xy=(epochs_sin[-1], val_acc_sin[-1]), 
                 xytext=(epochs_sin[-1]-3, val_acc_sin[-1]-5),
                 fontsize=10, color=COLOR_SIN_AUG)
    ax2.annotate(f'CON Aug: {val_acc_con[-1]:.1f}%', 
                 xy=(epochs_con[-1], val_acc_con[-1]), 
                 xytext=(epochs_con[-1]-3, val_acc_con[-1]+3),
                 fontsize=10, color=COLOR_CON_AUG)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'augmentation_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'augmentation_comparison.pdf', bbox_inches='tight')
    print(f"✓ Guardado: {OUTPUT_DIR / 'augmentation_comparison.png'}")
    plt.close()

def plot_minority_recall():
    """Figura 2: Comparación de recall por clase minoritaria"""
    
    # Datos de las 17 clases minoritarias (≤30 muestras)
    # Extraídos de los CSV
    data = {
        'Clase': [
            'Obfuscator.AD', 'Swizzor.gen!E', 'Alueron.gen!J', 'C2LOP.gen!g',
            'Rbot!gen', 'Dontovo.A', 'Autorun.K', 'Malex.gen!J', 'Lolyda.AA2',
            'Lolyda.AA3', 'Lolyda.AT', 'C2LOP.P', 'Skintrim.N', 'Adialer.C',
            'Swizzor.gen!I', 'Agent.FYI', 'Wintrim.BX'
        ],
        'Support': [30, 30, 29, 28, 26, 26, 21, 20, 20, 19, 19, 17, 17, 16, 13, 12, 10],
        'SIN_Aug': [100, 100, 100, 100, 100, 100, 100, 95, 100, 94.7, 100, 100, 100, 100, 100, 100, 100],
        'CON_Aug': [100, 90, 100, 100, 92.3, 100, 100, 95, 100, 94.7, 100, 94.1, 100, 100, 84.6, 100, 100]
    }
    
    df = pd.DataFrame(data)
    df['Delta'] = df['CON_Aug'] - df['SIN_Aug']
    
    # Ordenar por delta (más afectadas primero)
    df = df.sort_values('Delta', ascending=True)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df['SIN_Aug'], width, label='SIN Augmentation', 
                   color=COLOR_SIN_AUG, edgecolor='white', linewidth=0.7)
    bars2 = ax.bar(x + width/2, df['CON_Aug'], width, label='CON Augmentation', 
                   color=COLOR_CON_AUG, edgecolor='white', linewidth=0.7)
    
    # Línea horizontal en 100%
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Etiquetas
    ax.set_ylabel('Recall (%)')
    ax.set_title('Comparación de Recall por Clase Minoritaria (17 clases con ≤30 muestras)')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row['Clase']}\n({row['Support']})" for _, row in df.iterrows()], 
                       rotation=45, ha='right', fontsize=9)
    ax.legend(loc='lower right')
    ax.set_ylim(80, 105)
    
    # Añadir deltas sobre las barras afectadas
    for i, (_, row) in enumerate(df.iterrows()):
        if row['Delta'] < 0:
            ax.annotate(f"{row['Delta']:.1f} pp", 
                       xy=(i + width/2, row['CON_Aug']), 
                       xytext=(i + width/2, row['CON_Aug'] - 3),
                       ha='center', fontsize=8, color='darkred', fontweight='bold')
    
    # Añadir texto de resumen
    promedio_sin = df['SIN_Aug'].mean()
    promedio_con = df['CON_Aug'].mean()
    ax.text(0.02, 0.98, f'Promedio SIN Aug: {promedio_sin:.1f}%\nPromedio CON Aug: {promedio_con:.1f}%\nDiferencia: {promedio_con - promedio_sin:.1f} pp', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'minority_recall.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'minority_recall.pdf', bbox_inches='tight')
    print(f"✓ Guardado: {OUTPUT_DIR / 'minority_recall.png'}")
    plt.close()

def plot_delta_heatmap():
    """Figura 3 (opcional): Heatmap de diferencias"""
    
    data = {
        'Clase': [
            'Swizzor.gen!I', 'Swizzor.gen!E', 'Rbot!gen', 'C2LOP.P',
            'Malex.gen!J', 'Lolyda.AA3', 'Obfuscator.AD', 'Alueron.gen!J',
            'C2LOP.gen!g', 'Dontovo.A', 'Autorun.K', 'Lolyda.AA2',
            'Lolyda.AT', 'Skintrim.N', 'Adialer.C', 'Agent.FYI', 'Wintrim.BX'
        ],
        'Delta': [-15.4, -10.0, -7.7, -5.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    }
    
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#e74c3c' if d < 0 else '#2ecc71' for d in df['Delta']]
    bars = ax.barh(df['Clase'], df['Delta'], color=colors, edgecolor='white')
    
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.axvline(x=-15, color='gray', linestyle='--', alpha=0.5, label='Umbral H2 (+15 pp)')
    
    ax.set_xlabel('Diferencia en Recall (pp)')
    ax.set_title('Impacto de Data Augmentation por Clase Minoritaria')
    ax.set_xlim(-20, 5)
    
    # Añadir valores
    for bar, val in zip(bars, df['Delta']):
        if val != 0:
            ax.text(val - 0.5, bar.get_y() + bar.get_height()/2, 
                   f'{val:.1f}', va='center', ha='right', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'delta_impact.png', dpi=300, bbox_inches='tight')
    print(f"✓ Guardado: {OUTPUT_DIR / 'delta_impact.png'}")
    plt.close()

if __name__ == "__main__":
    print("Generando figuras para H2...")
    print("-" * 40)
    
    plot_training_curves()
    plot_minority_recall()
    plot_delta_heatmap()
    
    print("-" * 40)
    print(f"✓ Todas las figuras guardadas en: {OUTPUT_DIR}")
    print("\nRecuerda descomentar las líneas \\includegraphics en h2-figures.tex")