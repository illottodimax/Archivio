import tkinter as tk
from queue import Queue
from threading import Thread
from tkinter import filedialog, messagebox, ttk, scrolledtext
import requests
from itertools import product

# Definizione delle ruote del Lotto
URL_RUOTE = {
    'BA': 'https://raw.githubusercontent.com/Lottopython/Estractionx/refs/heads/main/BARI.txt',
    'CA': 'https://raw.githubusercontent.com/Lottopython/Estractionx/refs/heads/main/CAGLIARI.txt',
    'FI': 'https://raw.githubusercontent.com/Lottopython/Estractionx/refs/heads/main/FIRENZE.txt',
    'GE': 'https://raw.githubusercontent.com/Lottopython/Estractionx/refs/heads/main/GENOVA.txt',
    'MI': 'https://raw.githubusercontent.com/Lottopython/Estractionx/refs/heads/main/MILANO.txt',
    'NA': 'https://raw.githubusercontent.com/Lottopython/Estractionx/refs/heads/main/NAPOLI.txt',
    'PA': 'https://raw.githubusercontent.com/Lottopython/Estractionx/refs/heads/main/PALERMO.txt',
    'RO': 'https://raw.githubusercontent.com/Lottopython/Estractionx/refs/heads/main/ROMA.txt',
    'TO': 'https://raw.githubusercontent.com/Lottopython/Estractionx/refs/heads/main/TORINO.txt',
    'VE': 'https://raw.githubusercontent.com/Lottopython/Estractionx/refs/heads/main/VENEZIA.txt',
    'NZ': 'https://raw.githubusercontent.com/Lottopython/Estractionx/refs/heads/main/NAZIONALE.txt'
}

class SequenzaSpiaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Predittore Bit Lotto")
        self.setup_variables()
        self.create_ui()
        self.queue = Queue()
        self.check_queue()
        self.tabella_valori = []
        self.decimal_results = {}
        self.binary_filename = None
        self.selected_ruota = None

    def setup_variables(self):
        self.dati = []
        self.carrello = [None] * 7
        self.pattern_ricerca = []
        self.filename = None
        self.pattern_cache = {}
        self.soglia_var = tk.IntVar(value=50)
        self.bit_var = tk.IntVar(value=0)
        self.ruota_var = tk.StringVar(value="BA")

    def create_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Frame selezione ruota
        ruota_frame = ttk.LabelFrame(main_frame, text="Selezione Ruota", padding="5")
        ruota_frame.pack(fill=tk.X, pady=5)
        ttk.Label(ruota_frame, text="Ruota:").pack(side=tk.LEFT, padx=5)
        ruota_menu = ttk.OptionMenu(ruota_frame, self.ruota_var, "BA", *URL_RUOTE.keys(),
                                   command=self.update_ruota_label)
        ruota_menu.pack(side=tk.LEFT, padx=5)
        self.ruota_label = ttk.Label(ruota_frame, text="Ruota selezionata: Bari")
        self.ruota_label.pack(side=tk.LEFT, padx=5)

        # Anteprima binaria
        preview_frame = ttk.LabelFrame(main_frame, text="Anteprima Binaria", padding="5")
        preview_frame.pack(fill=tk.X, pady=5)
        self.binary_preview = scrolledtext.ScrolledText(preview_frame, height=5, width=50)
        self.binary_preview.pack(fill=tk.X, pady=5)

        # Conversione colonne
        self.conversion_frame = ttk.LabelFrame(main_frame, text="Conversione File", padding="5")
        self.conversion_frame.pack(fill=tk.X, pady=5)
        ttk.Button(self.conversion_frame, text="Carica e Converti Ruota",
                  command=self.carica_file).pack(side=tk.LEFT, padx=5)
        self.nome_file_label = ttk.Label(self.conversion_frame, text="Nessun dato caricato")
        self.nome_file_label.pack(side=tk.LEFT, padx=5)
        self.column_vars = []
        for i in range(5):
            var = tk.BooleanVar()
            self.column_vars.append(var)
            ttk.Checkbutton(self.conversion_frame, text=f"Col {i+1}",
                           variable=var).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.conversion_frame, text="Converti Colonne",
                  command=self.convert_selected_columns).pack(side=tk.LEFT, padx=5)

        # Analisi pattern
        self.analysis_frame = ttk.LabelFrame(main_frame, text="Analisi Pattern", padding="5")
        self.analysis_frame.pack(fill=tk.X, pady=5)
        ttk.Button(self.analysis_frame, text="Carica File Binario",
                  command=self.load_binary_file).pack(side=tk.LEFT, padx=5)
        self.binary_file_label = ttk.Label(self.analysis_frame, text="Nessun file binario caricato")
        self.binary_file_label.pack(side=tk.LEFT, padx=5)
        ttk.Label(self.analysis_frame, text="Righe:").pack(side=tk.LEFT, padx=5)
        self.righe_input = ttk.Entry(self.analysis_frame, width=5)
        self.righe_input.pack(side=tk.LEFT, padx=2)
        ttk.Label(self.analysis_frame, text="Colonne:").pack(side=tk.LEFT, padx=5)
        self.colonne_input = ttk.Entry(self.analysis_frame, width=5)
        self.colonne_input.pack(side=tk.LEFT, padx=2)
        ttk.Button(self.analysis_frame, text="Crea Pattern",
                  command=self.crea_tabella).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.analysis_frame, text="Carica Pattern",
                  command=self.carica_pattern).pack(side=tk.LEFT, padx=5)

        # Gestione carrello
        cart_frame = ttk.LabelFrame(main_frame, text="Gestione Carrello", padding="5")
        cart_frame.pack(fill=tk.X, pady=5)
        self.carrello_label = ttk.Label(cart_frame, text=self._format_carrello())
        self.carrello_label.pack(side=tk.LEFT, padx=5)
        ttk.Button(cart_frame, text="Svuota Carrello",
                  command=self.svuota_carrello).pack(side=tk.LEFT, padx=5)
        ttk.Button(cart_frame, text="Converti in Decimale",
                  command=self.esporta_carrello).pack(side=tk.LEFT, padx=5)

        # Tabella pattern
        self.table_frame = ttk.Frame(main_frame)
        self.table_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='determinate')
        self.progress.pack(fill=tk.X, pady=5)

        # Controllo pattern
        pattern_control_frame = ttk.LabelFrame(main_frame, text="Controllo Pattern", padding="5")
        pattern_control_frame.pack(fill=tk.X, pady=5)
        self.pattern_display = ttk.Label(pattern_control_frame, text="Pattern: Nessun pattern caricato")
        self.pattern_display.pack(side=tk.LEFT, padx=5)
        ttk.Button(pattern_control_frame, text="Analizza Dataset",
                  command=self.start_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(pattern_control_frame, text="Analizza Migliori Pattern",
                  command=self.start_best_pattern_analysis).pack(side=tk.LEFT, padx=5)

        # Selezione bit
        bit_frame = ttk.LabelFrame(main_frame, text="Selezione Bit", padding="5")
        bit_frame.pack(fill=tk.X, pady=5)
        ttk.Label(bit_frame, text="Posizione bit (0-6):").pack(side=tk.LEFT, padx=5)
        self.bit_position = ttk.Spinbox(bit_frame, from_=0, to=6, width=5)
        self.bit_position.pack(side=tk.LEFT, padx=5)
        self.bit_position.set(0)

        # Range analisi
        range_frame = ttk.LabelFrame(main_frame, text="Range Analisi", padding="5")
        range_frame.pack(fill=tk.X, pady=5)
        ttk.Label(range_frame, text="Ultime N estrazioni (0=tutte):").pack(side=tk.LEFT, padx=5)
        self.n_estrazioni = ttk.Entry(range_frame, width=6)
        self.n_estrazioni.pack(side=tk.LEFT, padx=5)
        self.n_estrazioni.insert(0, "0")

        # Risultati
        self.results_frame = ttk.LabelFrame(main_frame, text="Risultati Analisi", padding="5")
        self.results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        button_frame = ttk.Frame(self.results_frame)
        button_frame.pack(fill=tk.X, pady=5)
        ttk.Button(button_frame, text="Copia Risultati",
                  command=self.copy_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancella Risultati",
                  command=self.clear_results).pack(side=tk.LEFT, padx=5)
        self.results_text = scrolledtext.ScrolledText(self.results_frame, height=10, width=50)
        self.results_text.pack(fill=tk.BOTH, expand=True, pady=5)

        # Risultati decimali
        decimal_frame = ttk.LabelFrame(main_frame, text="Risultati Decimali per Colonna", padding="5")
        decimal_frame.pack(fill=tk.X, pady=5)
        self.decimal_text = scrolledtext.ScrolledText(decimal_frame, height=5, width=50)
        self.decimal_text.pack(fill=tk.X, pady=5)

    def update_ruota_label(self, value):
        self.selected_ruota = value
        self.ruota_label.config(text=f"Ruota selezionata: {self.get_ruota_name(value)}")

    def get_ruota_name(self, code):
        ruote_nomi = {
            'BA': 'Bari', 'CA': 'Cagliari', 'FI': 'Firenze', 'GE': 'Genova',
            'MI': 'Milano', 'NA': 'Napoli', 'PA': 'Palermo', 'RO': 'Roma',
            'TO': 'Torino', 'VE': 'Venezia', 'NZ': 'Nazionale'
        }
        return ruote_nomi.get(code, code)

    def carica_file(self):
        try:
            if not self.selected_ruota:
                self.selected_ruota = self.ruota_var.get()
            url = URL_RUOTE.get(self.selected_ruota)
            if not url:
                raise ValueError("Ruota non valida selezionata")
            response = requests.get(url)
            response.raise_for_status()
            self.filename = f"{self.selected_ruota}.txt"
            raw_lines = response.text.splitlines()
            self.dati = []
            for line in raw_lines:
                parts = line.strip().split()
                # MODIFICA CHIAVE: controlla almeno 7 elementi e i numeri validi sono da posizione 2 a 6
                if len(parts) >=7 and all(p.isdigit() and 1 <= int(p) <=90 for p in parts[2:7]):
                    numeri = parts[2:7]  # Estrai solo i 5 numeri validi
                    self.dati.append(" ".join(numeri))
                else:
                    print(f"Riga scartata: {line}")
            print(f"Righe valide caricate: {len(self.dati)}")
            self.nome_file_label.config(text=f"Ruota: {self.get_ruota_name(self.selected_ruota)}")
            # Converti colonne 1-5 (corrispondono ai numeri estratti)
            for i in range(1,6):
                output_file = f"b{i}.txt"
                self.convert_column_to_binary_file(i, output_file)
                print(f"Convertita colonna {i} in {output_file}")
            Thread(target=self.load_data, daemon=True).start()
        except Exception as e:
            self.queue.put(("error", f"Errore caricamento o conversione dati ruota: {str(e)}"))

    def load_data(self):
        try:
            print(f"Dati caricati per {self.selected_ruota}: {self.dati[:5]}...")
            self.queue.put(("update", f"Totale righe valide: {len(self.dati)} - Colonne convertite in b1.txt - b5.txt"))
            preview_text = '\n'.join(self.dati[:5]) + "\n..." if self.dati else "Nessun dato valido"
            self.binary_preview.delete('1.0', tk.END)
            self.binary_preview.insert('1.0', preview_text)
        except Exception as e:
            self.queue.put(("error", f"Errore elaborazione dati: {str(e)}"))

    def genera_report(self, file_binario, pattern, zero_count, one_count, binary_sequence, worst_bit, alternative_bit, decimal_value, position, next_bits=None):
        report = f"Per {file_binario}: {len(self.dati)} split con probabilità >= soglia.\n"
        report += "=== CONDIZIONE OTTIMALE TROVATA ===\n"
        report += f"[INFO] Ruota: {position}\n"
        report += f"[INFO] File binario: {file_binario}\n"
        report += f"[INFO] Pattern analizzato: {''.join(str(bit) for bit in pattern[0]) if isinstance(pattern, list) and pattern else pattern}\n"
        report += f"[INFO] Occorrenze '0': {zero_count}\n"
        report += f"[INFO] Occorrenze '1': {one_count}\n"
        report += f"[INFO] Sequenza binaria risultante: {binary_sequence} < bit 'peggiore' il {worst_bit + 1} -> alternativa bit: {alternative_bit} = {int(alternative_bit, 2)}\n"
        if next_bits:
            report += f"[INFO] Prossimi 3 bit: {''.join(map(str, next_bits))}\n"
        report += f"[INFO] Numero decimale: {decimal_value}\n"
        report += f"[INFO] ================================\n"
        self.results_text.insert(tk.END, report)
        self.decimal_text.insert(tk.END, f"Posizione: {position}, Decimale: {decimal_value}\n")

    def _format_carrello(self):
        return "Carrello: " + " ".join(str(b) if b is not None else "_" for b in self.carrello)

    def check_queue(self):
        while not self.queue.empty():
            msg = self.queue.get()
            if isinstance(msg, tuple):
                msg_type, content = msg
                if msg_type == "error":
                    messagebox.showerror("Errore", content)
                elif msg_type == "update":
                    self.nome_file_label.config(text=content)
            self.root.after(100, self.check_queue)

    def start_analysis(self):
        if not self.dati:
            messagebox.showwarning("Attenzione", "Caricare prima un file binario o una ruota")
            return
        if not self.pattern_ricerca:
            messagebox.showwarning("Attenzione", "Caricare prima un pattern")
            return
        Thread(target=self.analizza_dataset, daemon=True).start()

    def start_best_pattern_analysis(self):
        if not self.dati:
            messagebox.showwarning("Attenzione", "Caricare prima un file binario o una ruota")
            return
        Thread(target=self.analizza_migliori_pattern, daemon=True).start()

    def esporta_carrello(self):
        if None in self.carrello:
            messagebox.showwarning("Avviso", "Completare il carrello")
            return
        val = int("".join(map(str, self.carrello)), 2)
        messagebox.showinfo("Risultato", f"Valore decimale: {val}")
        bit_pos = self.binary_filename or self.selected_ruota
        n_estrazioni = self.n_estrazioni.get().strip()
        if n_estrazioni.isdigit():
            n_estrazioni = int(n_estrazioni)
        else:
            n_estrazioni = 0
        self.decimal_results[bit_pos] = (val, n_estrazioni)
        self.update_results_label()

    def crea_tabella(self):
        try:
            righe = int(self.righe_input.get())
            colonne = int(self.colonne_input.get())
            for widget in self.table_frame.winfo_children():
                widget.destroy()
            self.tabella_valori = []
            for i in range(righe):
                row = []
                for j in range(colonne):
                    btn = ttk.Button(self.table_frame, text="0", width=2,
                                    command=lambda r=i, c=j: self.toggle_bit(r, c))
                    btn.grid(row=i, column=j, padx=2, pady=2)
                    row.append(btn)
                self.tabella_valori.append(row)
            self.queue.put(("update", "Tabella pattern creata"))
        except ValueError:
            self.queue.put(("error", "Inserire valori numerici validi"))

    def toggle_bit(self, riga, colonna):
        button = self.tabella_valori[riga][colonna]
        current = button.cget("text")
        new_value = "1" if current == "0" else "0"
        button.config(text=new_value)

    def carica_pattern(self):
        try:
            pattern = []
            for row in self.tabella_valori:
                pattern_row = []
                for button in row:
                    pattern_row.append(int(button.cget("text")))
                pattern.append(pattern_row)
            self.pattern_ricerca = pattern
            pattern_str = "\n".join(" ".join(str(bit) for bit in row) for row in pattern)
            self.pattern_display.config(text=f"Pattern:\n{pattern_str}")
            messagebox.showinfo("Pattern Caricato", "Pattern pronto per l'analisi")
        except Exception as e:
            messagebox.showerror("Errore", f"Errore caricamento pattern: {str(e)}")

    def analizza_dataset(self):
        try:
            if not self.pattern_ricerca:
                messagebox.showwarning("Attenzione", "Caricare prima un pattern")
                return
            self.progress['value'] = 0
            reversed_data = list(reversed(self.dati))
            n = self.n_estrazioni.get().strip()
            if n and n.isdigit():
                n = int(n)
                if n > 0:
                    reversed_data = reversed_data[:n]
            print(f"Analisi limitata alle ultime {n} estrazioni" if n >0 else "Analisi su tutte le estrazioni")
            total_rows = len(reversed_data)
            if total_rows ==0:
                messagebox.showwarning("Attenzione", "Nessuna estrazione da analizzare.")
                return
            bit_pos = int(self.bit_position.get())
            zeros = 0
            ones = 0
            matches =0
            pattern_height = len(self.pattern_ricerca)
            pattern_width = len(self.pattern_ricerca[0]) if self.pattern_ricerca else 0
            for i in range(len(reversed_data) - pattern_height +1):
                match = True
                current_sequence = []
                for row_idx in range(pattern_height):
                    data_row = reversed_data[i + row_idx].split()
                    if not data_row:
                        match = False
                        break
                    current_sequence.append(reversed_data[i + row_idx])
                    if len(data_row)*7 < pattern_width:
                        match = False
                        break
                    binary_row = ''.join(data_row).replace('.', '')
                    for col_idx, pattern_bit in enumerate(self.pattern_ricerca[row_idx]):
                        if col_idx >= len(binary_row):
                            match = False
                            break
                        try:
                            if int(binary_row[col_idx]) != pattern_bit:
                                match = False
                                break
                        except ValueError as e:
                            print(f"Errore di conversione: {e} in riga {i+row_idx}, colonna {col_idx}")
                            match = False
                            break
                    if not match:
                        break
                if match:
                    matches +=1
                    next_row_idx = i + pattern_height
                    if next_row_idx < total_rows:
                        next_row = reversed_data[next_row_idx].split()
                        if next_row and bit_pos < len(''.join(next_row)):
                            binary_next_row = ''.join(next_row).replace('.', '')
                            worst_bit_index = bit_pos
                            alternative_bit = list(binary_next_row)
                            alternative_bit[bit_pos] = '1' if binary_next_row[bit_pos] == '0' else '0'
                            alternative_bit_str = ''.join(alternative_bit)
                            decimal_value = int(binary_next_row, 2)
                            if int(binary_next_row[bit_pos]) ==1:
                                ones +=1
                            else:
                                zeros +=1
                            next_bits = []
                            if next_row_idx +2 < total_rows:
                                for k in range(3):
                                    next_k_row = reversed_data[next_row_idx +k].split()
                                    if next_k_row and bit_pos < len(''.join(next_k_row)):
                                        next_bits.append(int(''.join(next_k_row).replace('.', '')[bit_pos]))
                            self.genera_report(self.binary_filename or self.selected_ruota, self.pattern_ricerca, zeros, ones, binary_next_row, worst_bit_index, alternative_bit_str, decimal_value, self.get_ruota_name(self.selected_ruota), next_bits if len(next_bits) ==3 else None)
                self.progress['value'] = (i/(total_rows - pattern_height +1))*100
                self.root.update_idletasks()
            if ones > zeros:
                self.carrello[bit_pos] = 1
            else:
                self.carrello[bit_pos] = 0
            self.carrello_label.config(text=self._format_carrello())
        except Exception as e:
            error_msg = f"Errore generico durante l'analisi: {str(e)}"
            print(f"ERRORE: {error_msg}")
            self.queue.put(("error", error_msg))
        finally:
            self.progress['value'] = 100
            self.root.update_idletasks()

    def analizza_migliori_pattern(self):
        try:
            self.progress['value'] =0
            reversed_data = list(reversed(self.dati))
            n = self.n_estrazioni.get().strip()
            if n and n.isdigit():
                n = int(n)
                if n>0:
                    reversed_data = reversed_data[:n]
            print(f"Analisi limitata alle ultime {n} estrazioni" if n>0 else "Analisi su tutte le estrazioni")
            total_rows = len(reversed_data)
            if total_rows ==0:
                messagebox.showwarning("Attenzione", "Nessuna estrazione da analizzare.")
                return
            bit_pos = int(self.bit_position.get())
            pattern_height =4
            pattern_width =7
            patterns = [list(map(int, p)) for p in product('01', repeat=pattern_height)]
            pattern_stats = {}
            for pattern in patterns:
                matches =0
                next_bits_list = []
                for i in range(len(reversed_data) - pattern_height -2):
                    match = True
                    for row_idx in range(pattern_height):
                        binary_row = reversed_data[i + row_idx].replace('.', '')
                        if len(binary_row) <= bit_pos or int(binary_row[bit_pos]) != pattern[row_idx]:
                            match =False
                            break
                    if match:
                        matches +=1
                        next_bits = []
                        for k in range(3):
                            next_row = reversed_data[i + pattern_height +k].replace('.', '')
                            if len(next_row) > bit_pos:
                                next_bits.append(int(next_row[bit_pos]))
                        if len(next_bits) ==3:
                            next_bits_list.append(next_bits)
                if matches>0:
                    pattern_stats[tuple(pattern)] = (matches, next_bits_list)
            top_patterns = sorted(pattern_stats.items(), key=lambda x: x[1][0], reverse=True)[:3]
            self.results_text.delete("1.0", tk.END)
            for pattern, (matches, next_bits_list) in top_patterns:
                zeros = sum(1 for bits in next_bits_list for b in bits if b==0)
                ones = sum(1 for bits in next_bits_list for b in bits if b==1)
                pattern_str = ''.join(map(str, pattern))
                predizione = []
                for i in range(3):
                    bit_zeros = sum(1 for bits in next_bits_list if bits[i]==0)
                    bit_ones = sum(1 for bits in next_bits_list if bits[i]==1)
                    predizione.append(1 if bit_ones>bit_zeros else 0)
                report = (f"Pattern: {pattern_str}\n"
                          f"Corrispondenze: {matches}\n"
                          f"Occorrenze totali - Zeri: {zeros}, Uni: {ones}\n"
                          f"Bit successivi trovati (primi 5):\n")
                for bits in next_bits_list[:5]:
                    report += f"  {''.join(map(str, bits))}\n"
                report += f"Predizione 3 bit: {''.join(map(str, predizione))}\n"
                self.results_text.insert(tk.END, report)
            self.progress['value'] =100
            self.root.update_idletasks()
        except Exception as e:
            error_msg = f"Errore generico durante l'analisi dei migliori pattern: {str(e)}"
            print(f"ERRORE: {error_msg}")
            self.queue.put(("error", error_msg))

    def convert_selected_columns(self):
        try:
            if not self.filename:
                messagebox.showwarning("Attenzione", "Caricare prima i dati di una ruota")
                return
            for i, var in enumerate(self.column_vars):
                if var.get():
                    output_file = f"b{i+1}.txt"
                    self.convert_column_to_binary_file(i+1, output_file)
        except Exception as e:
            messagebox.showerror("Errore", f"Errore nella conversione: {str(e)}")

    def convert_column_to_binary_file(self, column_index, output_file):
        try:
            transformed_lines = []
            for line in self.dati:
                numbers = line.strip().split()
                if column_index-1 < len(numbers):
                    num = numbers[column_index-1]
                    if num.isdigit() and 1 <= int(num) <=90:
                        binary = f"{int(num):07b}"
                        binary_with_dots = ".".join(binary)
                        transformed_lines.append(binary_with_dots)
            with open(output_file, 'w') as f:
                f.write('\n'.join(transformed_lines))
            print(f"File {output_file} creato con {len(transformed_lines)} righe")
        except Exception as e:
            print(f"Errore colonna {column_index}: {str(e)}")
            raise

    def load_binary_file(self):
        try:
            filename = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
            if filename:
                self.binary_filename = filename.split('/')[-1]
                self.binary_file_label.config(text=f"File binario: {self.binary_filename}")
                with open(filename, 'r') as file:
                    self.dati = [line.strip().replace('.', '') for line in file.readlines()]
                preview_text = '\n'.join(self.dati[:5]) + "\n..." if self.dati else "Nessun dato valido"
                self.binary_preview.delete('1.0', tk.END)
                self.binary_preview.insert('1.0', preview_text)
                self.queue.put(("update", f"File binario caricato: {len(self.dati)} righe"))
        except Exception as e:
            self.queue.put(("error", f"Errore caricamento file binario: {str(e)}"))

    def svuota_carrello(self):
        self.carrello = [None]*7
        self.carrello_label.config(text=self._format_carrello())

    def update_results_label(self):
        result_text = "Risultati Analisi:\n"
        for bit_pos, (decimal_value, n_estrazioni) in self.decimal_results.items():
            result_text += f"Decimale ricostruito: {decimal_value}\n"
            result_text += f"Rilevato {decimal_value} in posizione {bit_pos} con ultime {n_estrazioni} estrazioni\n"
        self.results_text.insert(tk.END, result_text + "\n")

    def copy_results(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.results_text.get("1.0", tk.END))
        messagebox.showinfo("Copiato", "Contenuto copiato negli appunti")

    def clear_results(self):
        self.results_text.delete("1.0", tk.END)
        self.decimal_text.delete("1.0", tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = SequenzaSpiaApp(root)
    root.mainloop()