# metodo_mensile.py
from collections import Counter

class MetodoMensile:
    def _fuori_90(self, numero):
        if numero > 90: return numero % 90 or 90
        return numero

    def _calcola_piramide(self, num1, num2):
        s = str(num1) + str(num2)
        while len(s) > 2:
            next_s = ""
            for i in range(len(s) - 1):
                somma = int(s[i]) + int(s[i+1])
                next_s += str((somma - 1) % 9 + 1)
            s = next_s
        return int(s)

    def calcola_previsione(self, estrazione_del_giorno, fisso=72):
        """
        Calcola la previsione. Ora accetta un 'fisso' opzionale.
        """
        try:
            napoli, palermo = estrazione_del_giorno['NA'], estrazione_del_giorno['PA']
            terzo_na, terzo_pa = napoli[2], palermo[2]
            quinto_na, quinto_pa = napoli[4], palermo[4]
        except (TypeError, IndexError, KeyError):
            return None

        somma_ambata = terzo_na + terzo_pa + fisso # Usa il fisso fornito
        ambata = self._fuori_90(somma_ambata)

        abbinamento1 = self._calcola_piramide(quinto_na, quinto_pa)
        abbinamento2 = 90 - abbinamento1

        return {
            'ambata': ambata, 'abbinamenti': sorted([abbinamento1, abbinamento2]),
            'ruote': ['NA', 'PA'], 'numeri_sorgente': {
                'NA': {'3': terzo_na, '5': quinto_na}, 'PA': {'3': terzo_pa, '5': quinto_pa}
            }
        }

class Correttore:
    def _fuori_90(self, numero):
        if numero > 90: return numero % 90 or 90
        return numero

    def trova_correttore_somma(self, casi_negativi):
        successi_per_correttore = Counter()
        for correttore_test in range(1, 91):
            for caso in casi_negativi:
                somma_base = caso['terzo_na'] + caso['terzo_pa']
                nuova_ambata = self._fuori_90(somma_base + correttore_test)
                if nuova_ambata in caso['esiti_reali']:
                    successi_per_correttore[correttore_test] += 1
        return successi_per_correttore.most_common()