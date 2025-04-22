import datetime
import json
import os
import hashlib
import sys
import uuid

# Import subprocess solo se su Windows per wmic
if os.name == 'nt':
    import subprocess

class LicenseSystem:
    """
    Gestisce la creazione, verifica e gestione delle licenze per l'applicazione.
    """
    def __init__(self, app_name="Numerical Empathy", license_filename="license.json"):
        """
        Inizializza il sistema di licenze.

        Args:
            app_name (str): Il nome dell'applicazione per cui è la licenza.
            license_filename (str): Il nome del file dove la licenza viene letta/scritta localmente.
        """
        self.app_name = app_name
        # Considera di salvare il file in una cartella dati utente per release
        self.license_file = license_filename

        # !!! ID SVILUPPATORE IMPOSTATO COME FORNITO !!!
        self.developer_machine_id = "4C4C4544-004D-5410-8057-C4C04F575633" # MANTENUTO COME INDICATO

    def get_machine_id(self):
        """
        Genera un ID macchina univoco basato su identificatori hardware stabili.

        Returns:
            str: Un hash MD5 rappresentante l'ID univoco della macchina.
                 Restituisce una stringa contenente 'error' se non riesce a determinare un ID affidabile.
        """
        machine_id_raw = ""
        try:
            if os.name == "nt":  # Windows
                # --- NUOVO CODICE (più robusto per Windows) ---
                try:
                    startupinfo = subprocess.STARTUPINFO()
                    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    startupinfo.wShowWindow = subprocess.SW_HIDE
                    result = subprocess.check_output(
                        "wmic csproduct get uuid",
                        universal_newlines=True, # Importante per decodificare l'output come testo
                        stderr=subprocess.PIPE,  # Cattura stderr invece di ignorarlo
                        startupinfo=startupinfo,
                        timeout=10 # Aggiungi un timeout per sicurezza
                    )

                    machine_id_raw = "" # Inizializza vuoto
                    lines = result.splitlines() # Usa splitlines() che gestisce meglio \n e \r\n

                    for line in lines:
                        cleaned_line = line.strip()
                        # Cerca una linea che assomigli a un UUID (lunga 36 caratteri con 4 trattini)
                        if len(cleaned_line) == 36 and cleaned_line.count('-') == 4:
                            try:
                                uuid.UUID(cleaned_line) # Valida il formato UUID
                                machine_id_raw = cleaned_line
                                break # Trovato l'UUID, esci dal ciclo
                            except ValueError:
                                continue # Non era un UUID valido, continua a cercare

                    # Se dopo il ciclo non abbiamo trovato un UUID valido nell'output
                    if not machine_id_raw:
                         print("Error: Could not parse UUID from wmic output.")
                         print(f"WMIC Raw Output:\n---\n{result}\n---")
                         return f"error_parsing_wmic_output_{uuid.uuid4()}"

                except subprocess.CalledProcessError as e:
                     print(f"Error executing wmic: {e}")
                     print(f"Stderr: {e.stderr}") # Stampa l'errore da wmic
                     return f"error_executing_wmic_{uuid.uuid4()}"
                except subprocess.TimeoutExpired:
                     print("Error: Timeout expired while executing wmic.")
                     return f"error_wmic_timeout_{uuid.uuid4()}"
                except FileNotFoundError:
                     print("Error: 'wmic' command not found. Is it in the system PATH?")
                     return f"error_wmic_not_found_{uuid.uuid4()}"
                # --- FINE NUOVO CODICE ---

            elif sys.platform == "darwin": # MacOS
                # Codice per MacOS (invariato)
                 result = subprocess.check_output(
                     ["ioreg", "-d2", "-c", "IOPlatformExpertDevice"],
                     universal_newlines=True,
                     stderr=subprocess.DEVNULL
                 ).strip()
                 for line in result.split('\n'):
                     if "IOPlatformUUID" in line:
                         machine_id_raw = line.split('=')[-1].strip().strip('"')
                         break

            else:  # Linux (e altri Unix-like)
                # Codice per Linux (invariato)
                found = False
                for path in ["/etc/machine-id", "/var/lib/dbus/machine-id"]:
                    try:
                        with open(path, "r") as f:
                            machine_id_raw = f.read().strip()
                            if machine_id_raw:
                                found = True
                                break
                    except FileNotFoundError:
                        continue
                    except Exception as e_read:
                         print(f"Warning: Could not read {path}: {e_read}")
                         continue

                if not found:
                    node = uuid.getnode()
                    if node is not None and node != 0 and node != 1:
                         mac_address = uuid.UUID(int=node).hex[-12:]
                         machine_id_raw = mac_address
                         print("Warning: Using MAC address as machine ID fallback (less stable).")
                    else:
                         import socket
                         hostname = socket.gethostname()
                         if hostname:
                              machine_id_raw = hostname
                              print("Warning: Using hostname as machine ID fallback (least stable).")

            # Se dopo tutti i tentativi non abbiamo ancora un ID valido
            if not machine_id_raw:
                 print("Error: Could not determine a reliable machine ID.")
                 return f"error_cannot_get_id_{uuid.uuid4()}"

            # --- MODIFICA IMPORTANTE: Controlla se il developer ID è l'UUID grezzo o l'hash MD5 ---
            # Il tuo developer ID sembra un UUID grezzo, quindi confrontiamo quello prima di fare l'hash
            if machine_id_raw.upper() == self.developer_machine_id.upper():
                # Se l'ID grezzo corrisponde al developer ID, restituiamo direttamente il developer ID
                # per far funzionare il bypass in check_license(), che si aspetta l'hash MD5.
                # Quindi, applichiamo l'hash MD5 al developer ID stesso.
                # Questo assume che developer_machine_id sia l'UUID grezzo e non l'hash.
                # Se developer_machine_id fosse già l'hash, questa logica andrebbe cambiata.
                print("Debug: Raw machine ID matches developer UUID. Returning developer hash.")
                return hashlib.md5(self.developer_machine_id.encode('utf-8')).hexdigest()
            else:
                # Altrimenti, per tutti gli altri, calcola l'hash MD5 dell'ID grezzo ottenuto
                return hashlib.md5(machine_id_raw.encode('utf-8')).hexdigest()

        except Exception as e:
            print(f"Fatal Error getting machine ID component: {e}")
            return f"error_exception_{uuid.uuid4()}"

    def check_license(self):
        """
        Verifica la validità della licenza memorizzata nel file locale.

        Returns:
            tuple: (bool is_valid, str message)
        """
        try:
            # --- MODIFICA IMPORTANTE ---
            # Ora confrontiamo l'hash MD5 del developer ID, perché get_machine_id restituirà
            # l'hash md5 del developer ID se il raw ID corrisponde.
            developer_machine_id_hashed = hashlib.md5(self.developer_machine_id.encode('utf-8')).hexdigest()
            current_machine_id_hashed = self.get_machine_id() # Ottiene l'hash MD5

            # Gestione errore se non si può ottenere l'ID macchina attuale
            if "error" in current_machine_id_hashed:
                 return False, f"Errore critico: Impossibile verificare l'ID di questa macchina ({current_machine_id_hashed}). Contattare supporto."

            # Bypass per la macchina dello sviluppatore (confronta gli hash MD5)
            # print(f"Debug Check: Current HASH = {current_machine_id_hashed}")
            # print(f"Debug Check: Dev HASH = {developer_machine_id_hashed}")
            if current_machine_id_hashed == developer_machine_id_hashed:
                return True, "Licenza sviluppatore attiva (accesso illimitato)"

            # Verifica standard per utenti normali
            if not os.path.exists(self.license_file):
                return False, f"File di licenza '{self.license_file}' non trovato."

            # Leggi il file di licenza
            with open(self.license_file, "r") as f:
                try:
                    license_data = json.load(f)
                except json.JSONDecodeError:
                    return False, "Errore: File di licenza corrotto o in formato non valido."

            # Verifiche Obbligatorie sul Contenuto
            required_keys = ["expiry_date", "license_key", "app_name"]
            if not all(key in license_data for key in required_keys):
                 return False, "Errore: Formato file di licenza non valido (mancano dati essenziali)."
            if license_data.get("app_name") != self.app_name:
                 return False, f"Errore: Licenza non valida per questa applicazione ('{self.app_name}')."

            # Verifica data di scadenza
            try:
                expiry_date_str = license_data["expiry_date"]
                expiry_date = datetime.datetime.strptime(expiry_date_str, "%Y-%m-%d")
                expiry_datetime = expiry_date.replace(hour=23, minute=59, second=59)
            except ValueError:
                return False, "Errore: Formato data di scadenza ('expiry_date') non valido nel file."

            current_datetime = datetime.datetime.now()

            if current_datetime > expiry_datetime:
                days_expired = (current_datetime.date() - expiry_date.date()).days
                return False, f"Licenza scaduta il {expiry_date_str} (da {days_expired} giorni)"

            # Verifica Machine ID (hash) se la licenza è vincolata
            license_machine_id_hashed = license_data.get("machine_id") # Assume che questo sia l'hash MD5
            if license_machine_id_hashed:
                if license_machine_id_hashed != current_machine_id_hashed:
                    # print(f"Debug Check: Current HASH = {current_machine_id_hashed}")
                    # print(f"Debug Check: License HASH = {license_machine_id_hashed}")
                    return False, "Licenza non valida per questo computer (ID non corrispondente)."

            # Licenza Valida
            days_remaining = (expiry_date.date() - current_datetime.date()).days
            is_trial = license_data.get("is_trial", False)
            license_type = "Licenza di Prova" if is_trial else "Licenza Completa"
            binding_info = " (legata a questa macchina)" if license_machine_id_hashed else " (non legata alla macchina)"

            return True, f"{license_type}{binding_info} - Attiva fino al {expiry_date_str} (ancora {days_remaining} giorni)"

        except FileNotFoundError:
             return False, f"File di licenza '{self.license_file}' non trovato."
        except PermissionError:
             return False, f"Errore: Permessi insufficienti per leggere il file di licenza '{self.license_file}'."
        except Exception as e:
            print(f"Errore imprevisto durante la verifica della licenza: {type(e).__name__} - {e}")
            return False, f"Errore imprevisto durante la verifica della licenza. Contattare supporto."


    def create_license(self, expiry_days=5, bind_to_machine=True, is_trial=True):
        """
        Crea un file di licenza LOCALMENTE (es. per attivare una prova).
        SOVRASCRIVE il file di licenza esistente.
        """
        try:
            expiry_date = datetime.datetime.now() + datetime.timedelta(days=expiry_days)
            license_data = {
                "app_name": self.app_name,
                "license_key": hashlib.md5(os.urandom(16)).hexdigest(),
                "expiry_date": expiry_date.strftime("%Y-%m-%d"),
                "creation_date": datetime.datetime.now().strftime("%Y-%m-%d"),
                "is_trial": is_trial,
                "machine_id": None
            }

            if bind_to_machine:
                current_machine_id_hashed = self.get_machine_id() # Ottiene l'hash MD5
                if "error" in current_machine_id_hashed:
                    return False, f"Errore: Impossibile ottenere l'ID macchina per legare la licenza ({current_machine_id_hashed})."
                license_data["machine_id"] = current_machine_id_hashed # Salva l'hash MD5

            with open(self.license_file, "w") as f:
                json.dump(license_data, f, indent=4)

            return True, license_data

        except PermissionError:
            return False, f"Errore: Permessi insufficienti per scrivere il file di licenza '{self.license_file}'."
        except Exception as e:
             print(f"Errore durante la creazione della licenza locale: {type(e).__name__} - {e}")
             return False, f"Errore imprevisto durante la creazione della licenza locale."


    def can_create_trial(self):
        """
        Controlla se è possibile generare una nuova licenza di prova.
        (Logica invariata)
        """
        if not os.path.exists(self.license_file):
            return True
        try:
            with open(self.license_file, "r") as f:
                license_data = json.load(f)
                required_keys = ["expiry_date", "license_key"]
                if not all(key in license_data for key in required_keys):
                     print("Warning: Existing license file format is invalid. Allowing trial creation attempt.")
                     return True
                return False
        except (json.JSONDecodeError, PermissionError) as e:
            print(f"Warning: Existing license file is unreadable ({e}). Allowing trial creation attempt.")
            return True
        except FileNotFoundError:
             return True
        except Exception as e_gen:
             print(f"Warning: Unexpected error checking existing license ({e_gen}). Allowing trial creation attempt.")
             return True


    def create_license_for_machine(self, machine_id_hashed_to_bind, expiry_days=30, is_trial=False):
        """
        Genera i DATI di licenza per una macchina specifica (uso per Admin Tool).
        Questa funzione NON salva/modifica il file di licenza locale.
        Restituisce solo il dizionario con i dati della licenza.

        Args:
            machine_id_hashed_to_bind (str): L'hash MD5 dell'ID macchina target (ottenuto dall'utente).
            expiry_days (int): Giorni di validità della licenza da generare.
            is_trial (bool): Se True, marca la licenza generata come 'di prova'.
        """
        # Validazione base dell'ID macchina fornito (dovrebbe essere un hash MD5 di 32 caratteri)
        if not isinstance(machine_id_hashed_to_bind, str) or len(machine_id_hashed_to_bind) != 32:
            try:
                # Verifica se è esadecimale valido
                int(machine_id_hashed_to_bind, 16)
            except (ValueError, TypeError):
                print(f"Errore: Formato Machine ID fornito non valido: '{machine_id_hashed_to_bind}'. Deve essere un hash MD5 di 32 caratteri esadecimali.")
                return None

        try:
            expiry_date = datetime.datetime.now() + datetime.timedelta(days=expiry_days)
            license_data = {
                "app_name": self.app_name,
                "license_key": hashlib.md5(os.urandom(16)).hexdigest(),
                "expiry_date": expiry_date.strftime("%Y-%m-%d"),
                "creation_date": datetime.datetime.now().strftime("%Y-%m-%d"),
                "is_trial": is_trial,
                "machine_id": machine_id_hashed_to_bind # Salva l'hash MD5 fornito
            }
            # NESSUNA SCRITTURA SU FILE LOCALE QUI
            return license_data
        except Exception as e:
            print(f"Errore durante la generazione dei dati di licenza per MID HASH {machine_id_hashed_to_bind}: {type(e).__name__} - {e}")
            return None


# ----- Esempio di utilizzo (per testare la classe) -----
if __name__ == "__main__":
    print("Test LicenseSystem...")
    # Usa un nome app diverso per il test per non sovrascrivere la licenza reale
    ls = LicenseSystem(app_name="TestApp", license_filename="test_license.json")

    print("\n--- Ottenimento Machine ID ---")
    my_id_hash = ls.get_machine_id()
    print(f"My Machine ID HASH: {my_id_hash}")
    print(f"(Developer Raw UUID impostato: {ls.developer_machine_id})")
    # Calcola l'hash del developer ID per confronto
    dev_id_hash = hashlib.md5(ls.developer_machine_id.encode('utf-8')).hexdigest()
    print(f"(Developer ID HASH: {dev_id_hash})")

    if my_id_hash == dev_id_hash:
        print("-> Questa macchina è riconosciuta come macchina sviluppatore (HASH corrispondente).")
    else:
        print("-> Questa macchina NON è riconosciuta come macchina sviluppatore.")


    print("\n--- Verifica Licenza Iniziale ---")
    # Rimuovi il file di test se esiste per una verifica pulita
    if os.path.exists(ls.license_file):
        os.remove(ls.license_file)
        print(f"File di test rimosso: {ls.license_file}")

    is_valid, message = ls.check_license()
    print(f"Stato iniziale: Valido={is_valid}, Messaggio='{message}'")

    print("\n--- Controllo Creazione Trial ---")
    can_trial = ls.can_create_trial()
    print(f"Può creare una licenza di prova? {can_trial}")

    if can_trial:
        print("\n--- Creazione Licenza di Prova Locale (7 giorni, legata) ---")
        success, data_or_msg = ls.create_license(expiry_days=7, bind_to_machine=True, is_trial=True)
        if success:
            print("Licenza di prova creata con successo.")
            is_valid, message = ls.check_license()
            print(f"Nuovo stato: Valido={is_valid}, Messaggio='{message}'")
        else:
            print(f"Errore creazione prova: {data_or_msg}")

    print("\n--- Tentativo di ri-creare Trial ---")
    can_trial_again = ls.can_create_trial()
    print(f"Può creare un'altra prova? {can_trial_again}") # Dovrebbe essere False ora

    print("\n--- Generazione Dati Licenza per Admin Tool (per HASH fittizio) ---")
    fake_user_id_hash = hashlib.md5("some_user_specific_string_that_gives_hash".encode()).hexdigest()
    print(f"Genero licenza per HASH utente fittizio: {fake_user_id_hash}")
    admin_data = ls.create_license_for_machine(fake_user_id_hash, expiry_days=365, is_trial=False)
    if admin_data:
        print("Dati generati per l'admin tool (salvare in license_userXYZ.json):")
        print(json.dumps(admin_data, indent=4))
    else:
        print("Errore nella generazione dei dati per l'admin tool.")

    # Pulizia opzionale del file di test
    # if os.path.exists(ls.license_file):
    #     print(f"\nRimozione file di test: {ls.license_file}")
    #     #os.remove(ls.license_file)