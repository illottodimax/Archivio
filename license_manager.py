# license_manager.py (file separato)
import tkinter as tk
from tkinter import messagebox, ttk
import os
import hashlib # Necessario per calcolare l'hash MD5
import uuid    # Necessario per validare l'UUID (opzionale ma consigliato)

# Assicurati che license_system.py sia importabile
try:
    from license_system import LicenseSystem
except ImportError:
    messagebox.showerror("Errore Critico", "File 'license_system.py' non trovato.\nAssicurati che sia nella stessa cartella di questo script.")
    exit() # Non si può continuare senza il sistema di licenze

def calculate_md5(input_string):
    """Calcola l'hash MD5 di una stringa."""
    return hashlib.md5(input_string.encode('utf-8')).hexdigest()

def renew_license():
    """Ottiene l'input, calcola l'hash se necessario, e crea la licenza."""
    raw_input_id = entry_machine_id.get().strip()
    days_str = entry_days.get().strip()
    machine_id_hash = None # Qui memorizzeremo l'hash MD5 finale

    # --- Validazione Input Giorni ---
    try:
        days = int(days_str)
        if days < 1:
            messagebox.showerror("Errore Input", "Il numero di giorni deve essere positivo.")
            return
    except ValueError:
        messagebox.showerror("Errore Input", "Inserisci un numero valido per i giorni.")
        return

    # --- Validazione e Processamento Input ID Macchina ---
    if not raw_input_id:
        messagebox.showerror("Errore Input", "Inserisci l'UUID o l'Hash MD5 della macchina utente.")
        return

    if len(raw_input_id) == 36 and raw_input_id.count('-') == 4:
        # Sembra un UUID grezzo (formato XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX)
        try:
            # Opzionale ma consigliato: Valida che sia un UUID ben formato
            uuid.UUID(raw_input_id)
            # Calcola l'hash MD5 dall'UUID grezzo
            machine_id_hash = calculate_md5(raw_input_id)
            print(f"INFO: Rilevato UUID grezzo. Hash MD5 calcolato: {machine_id_hash}") # Log/Debug
        except ValueError:
             messagebox.showerror("Errore Formato", "L'ID inserito sembra un UUID ma il suo formato non è valido.")
             return
        except Exception as e_hash:
             messagebox.showerror("Errore Calcolo Hash", f"Errore durante il calcolo dell'hash MD5: {e_hash}")
             return

    elif len(raw_input_id) == 32:
        # Sembra un hash MD5 (32 caratteri)
        try:
            # Verifica se è composto solo da caratteri esadecimali
            int(raw_input_id, 16)
            machine_id_hash = raw_input_id # È già l'hash, usalo direttamente
            print(f"INFO: Rilevato potenziale Hash MD5: {machine_id_hash}") # Log/Debug
        except ValueError:
            messagebox.showerror("Errore Formato", "L'ID inserito sembra un hash MD5 ma contiene caratteri non esadecimali.")
            return
    else:
        # Formato non riconosciuto
        messagebox.showerror("Errore Formato", "Formato Machine ID non riconosciuto.\nInserisci l'UUID grezzo (ottenuto da 'wmic csproduct get uuid') o l'hash MD5 (32 caratteri).")
        return

    # A questo punto, machine_id_hash DOVREBBE contenere l'hash MD5 corretto
    if not machine_id_hash:
        messagebox.showerror("Errore Interno", "Impossibile determinare l'hash MD5 finale per la licenza.")
        return

    # --- Creazione Licenza usando l'HASH MD5 ---
    try:
        license_system = LicenseSystem()
        # Chiama create_license_for_machine passando l'HASH MD5
        license_data = license_system.create_license_for_machine(machine_id_hash, expiry_days=days)

        # Controlla se la funzione ha restituito None (indicando un errore interno)
        if license_data is None:
             messagebox.showerror("Errore Creazione", "La creazione dei dati di licenza è fallita.\nControlla che l'hash MD5 sia corretto e che non ci siano altri errori.")
             return

        # Salva la licenza in un file che l'utente può importare
        # Usa l'HASH MD5 per il nome del file per coerenza
        output_file = f"license_{machine_id_hash[:8]}.json"
        with open(output_file, "w") as f:
            # Nota: L'import di json qui è ok ma potrebbe essere fatto all'inizio
            import json
            json.dump(license_data, f, indent=4)

        messagebox.showinfo(
            "Licenza Creata",
            f"Licenza creata con successo per {days} giorni.\n"
            f"Machine ID Hash usato: {machine_id_hash}\n" # Mostra l'hash usato
            f"File licenza: {output_file}\n"
            f"Scadenza: {license_data['expiry_date']}"
        )
    except Exception as e:
        messagebox.showerror("Errore", f"Errore imprevisto durante la creazione/salvataggio della licenza: {e}")


# --- Il resto dell'UI (Login, Frame, etc.) rimane invariato ---

# Verifica amministratore (invariato)
def check_admin():
    password = password_entry.get()
    if password != "310763Massimo$":  # Sostituisci con la tua password
        messagebox.showerror("Errore", "Password non valida")
        return

    login_frame.pack_forget()
    main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

# UI (invariato, ma potresti cambiare la label)
root = tk.Tk()
root.title("License Manager (Admin)")
root.geometry("500x400") # Leggermente più largo per la label

# Frame di login (invariato)
login_frame = tk.Frame(root, padx=20, pady=20)
login_frame.pack(fill=tk.BOTH, expand=True)
tk.Label(login_frame, text="Password Amministratore:").pack(pady=10)
password_entry = tk.Entry(login_frame, show="*", width=30)
password_entry.pack(pady=10)
tk.Button(login_frame, text="Accedi", command=check_admin).pack(pady=20)

# Frame principale (nascosto inizialmente)
main_frame = tk.Frame(root)

# Sezione creazione licenza
license_frame = tk.LabelFrame(main_frame, text="Crea/Rinnova Licenza", padx=10, pady=10)
license_frame.pack(pady=10, fill=tk.X)

# Label modificata per chiarezza
tk.Label(license_frame, text="Machine ID Utente (UUID o Hash MD5):").grid(row=0, column=0, padx=5, pady=5, sticky="e")
entry_machine_id = tk.Entry(license_frame, width=40)
entry_machine_id.grid(row=0, column=1, padx=5, pady=5, sticky="w")

tk.Label(license_frame, text="Giorni di validità:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
entry_days = tk.Entry(license_frame, width=10)
entry_days.insert(0, "30")
entry_days.grid(row=1, column=1, padx=5, pady=5, sticky="w")

# Pulsante Crea Licenza chiama la NUOVA funzione renew_license
tk.Button(license_frame, text="Crea Licenza", command=renew_license, bg="#4CAF50", fg="white").grid(row=2, column=0, columnspan=2, pady=10)

# Nota: La sezione commentata con create_license_for_machine non è più necessaria qui
# dato che il metodo è in license_system.py

root.mainloop()