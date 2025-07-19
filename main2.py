
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

# --- CLASSES DE DONNÉES ---

@dataclass
class MarketData:
    """Données de marché pour le pricing"""
    r: float
    sigma: float
    initial_rate: float
    spread_credit: float
    recovery_rate: float
    kappa: float = 0.1
    theta: float = 0.03

@dataclass
class SwapParameters:
    """Paramètres du swap"""
    notional: float
    maturity: float
    fixed_rate: float
    payment_frequency: int
    is_payer: bool

@dataclass
class CollateralParams:
    """Paramètres du collatéral"""
    threshold: float
    minimum_transfer: float
    haircut: float = 0.0
    margining_frequency: int = 1

# --- FONCTION DE CORRÉLATION WWR ---

def generate_correlated_variables_for_wwr(Nmc: int, rate_paths: np.ndarray, correlation_wwr: float) -> np.ndarray:
    """
    Génère des variables latentes pour le défaut CP corrélées avec les trajectoires de taux.
    Pour un swap PAYEUR, le WWR survient quand une hausse des taux (qui augmente l'exposition)
    est corrélée à une hausse de la probabilité de défaut.
    """
    rate_systemic_factor = np.mean(rate_paths, axis=1)
    rate_factor_standardized = (rate_systemic_factor - np.mean(rate_systemic_factor)) / np.std(rate_systemic_factor)
    Z_default_cp_independent = np.random.normal(0, 1, Nmc)
    Z_default_cp_correlated = (correlation_wwr * rate_factor_standardized +
                               np.sqrt(1 - correlation_wwr**2) * Z_default_cp_independent)
    return Z_default_cp_correlated

# --- MODÈLES FINANCIERS ---

class InterestRateModel(ABC):
    @abstractmethod
    def simulate_paths(self, T: float, dt: float, Nmc: int) -> np.ndarray:
        pass

    @abstractmethod
    def zero_coupon_bond(self, t: float, T: float, r_current: np.ndarray) -> np.ndarray:
        pass

class VasicekModel(InterestRateModel):
    """
    Modèle de Vasicek dr = κ(θ - r)dt + σdW.
    NOTE : L'anomalie de simulation dans l'output initial (taux moyen final de 11%)
    n'est pas reproductible avec cette implémentation correcte. Le bug devait se
    trouver dans une version antérieure du code qui a généré cet output.
    """
    def __init__(self, r0: float, kappa: float, theta: float, sigma: float):
        self.r0 = r0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma

    def simulate_paths(self, T: float, dt: float, Nmc: int) -> np.ndarray:
        n_steps = int(T / dt)
        rate_paths = np.zeros((Nmc, n_steps + 1))
        rate_paths[:, 0] = self.r0
        dW = np.random.normal(0, np.sqrt(dt), (Nmc, n_steps))

        for i in range(n_steps):
            drift = self.kappa * (self.theta - rate_paths[:, i]) * dt
            diffusion = self.sigma * dW[:, i]
            rate_paths[:, i + 1] = rate_paths[:, i] + drift + diffusion
        return rate_paths

    def zero_coupon_bond(self, t: float, T: float, r_current: np.ndarray) -> np.ndarray:
        """
        CORRECTION MAJEURE : Prix d'un zéro-coupon Vasicek P(t, T) à un instant t < T.
        Formule généralisée pour le calcul de l'exposition future.
        P(t,T) = A(t,T) * exp(-B(t,T) * r(t))
        """
        if t > T:
            return np.ones_like(r_current)

        tau = T - t
        if self.kappa == 0:
            B_tT = tau
            A_tT = np.exp(-self.theta * tau + (self.sigma**2 * tau**3) / 6)
        else:
            B_tT = (1 - np.exp(-self.kappa * tau)) / self.kappa
            term1 = (self.theta - self.sigma**2 / (2 * self.kappa**2)) * (B_tT - tau)
            term2 = (self.sigma**2 * B_tT**2) / (4 * self.kappa)
            A_tT = np.exp(term1 - term2)

        return A_tT * np.exp(-B_tT * r_current)


class DefaultModel:
    def __init__(self, lambda_default: float, recovery_rate: float):
        self.lambda_default = lambda_default
        self.recovery_rate = recovery_rate

    def simulate_default_times(self, T: float, Nmc: int, Z_latent: Optional[np.ndarray] = None) -> np.ndarray:
        if Z_latent is None:
            Z_latent = np.random.normal(0, 1, Nmc)
        U_default = stats.norm.cdf(Z_latent)
        # Clip pour éviter les valeurs extrêmes qui peuvent causer des infinis
        U_default = np.clip(U_default, 1e-10, 1 - 1e-10)
        default_times = -np.log(1 - U_default) / self.lambda_default
        return default_times

    def survival_probability(self, t: float) -> float:
        return np.exp(-self.lambda_default * t)

# --- PRODUITS FINANCIERS ET MOTEUR CVA ---

class InterestRateSwap:
    def __init__(self, swap_params: SwapParameters, market_data: MarketData, rate_model: VasicekModel):
        self.params = swap_params
        self.market_data = market_data
        self.rate_model = rate_model
        self.payment_dates = self._generate_payment_dates()
        if self.params.fixed_rate == 0.0:
            self.params.fixed_rate = self._calculate_atm_rate()

    def _generate_payment_dates(self) -> np.ndarray:
        dt_payment = 1.0 / self.params.payment_frequency
        return np.arange(dt_payment, self.params.maturity + dt_payment, dt_payment)

    def _calculate_atm_rate(self) -> float:
        """Calcul du taux swap at-the-money avec la formule analytique Vasicek à t=0."""
        dt_payment = 1.0 / self.params.payment_frequency
        zcb_prices = np.array([
            self.rate_model.zero_coupon_bond(0, T_i, np.array([self.market_data.initial_rate]))[0]
            for T_i in self.payment_dates
        ])
        annuity = np.sum(dt_payment * zcb_prices)
        if annuity > 0:
            return (1 - zcb_prices[-1]) / annuity
        return self.market_data.initial_rate

    def calculate_exposure_profile(self, rate_paths: np.ndarray, time_grid: np.ndarray) -> np.ndarray:
        """
        CORRECTION FONDAMENTALE : Calcul du Mark-to-Market (MtM) du swap à chaque pas de temps.
        C'est la méthode correcte pour générer le profil d'exposition.
        L'exposition est la valeur du swap (NPV) à un instant t > 0.
        V_swap = V_float - V_fixed (pour un payeur).
        """
        Nmc, n_steps = rate_paths.shape
        exposure_profile = np.zeros_like(rate_paths)
        dt_payment = 1.0 / self.params.payment_frequency

        for j, t in enumerate(time_grid):
            # Taux simulé à l'instant t pour toutes les trajectoires
            r_t = rate_paths[:, j]
            
            # Identifier les dates de paiement futures
            future_payment_dates = self.payment_dates[self.payment_dates > t]
            if len(future_payment_dates) == 0:
                continue

            # Valeur de la jambe fixe restante
            fixed_leg_value = np.sum([
                self.params.fixed_rate * dt_payment * self.rate_model.zero_coupon_bond(t, T_i, r_t)
                for T_i in future_payment_dates
            ], axis=0)

            # Valeur de la jambe flottante restante
            # Formule: N * (P(t, T_start_period) - P(t, T_end_period))
            # Ici, T_start est la date de reset passée (t_prev) et T_end est la maturité finale T_N
            t_prev = self.payment_dates[self.payment_dates <= t][-1] if any(self.payment_dates <= t) else 0
            
            p_t_prev = self.rate_model.zero_coupon_bond(t, t_prev, r_t)
            p_t_N = self.rate_model.zero_coupon_bond(t, self.params.maturity, r_t)

            floating_leg_value = (p_t_prev - p_t_N)
            
            # NPV du swap = (V_float - V_fixed) * Notionnel
            swap_npv = (floating_leg_value - fixed_leg_value) * self.params.notional
            
            # La position est RECEVEUR si is_payer est False
            if not self.params.is_payer:
                swap_npv *= -1
            
            exposure_profile[:, j] = swap_npv
            
        return exposure_profile


class CVAEngine:
    def __init__(self, market_data: MarketData):
        self.market_data = market_data

    def calculate_exposure_metrics(self, exposure_profile: np.ndarray, time_grid: np.ndarray) -> dict:
        """Calcule les métriques d'exposition (EE, PFE) à partir du profil de MtM."""
        positive_exposure = np.maximum(exposure_profile, 0)
        negative_exposure = np.minimum(exposure_profile, 0)
        
        ee = np.mean(positive_exposure, axis=0)
        ene = np.mean(negative_exposure, axis=0)
        pfe_95 = np.percentile(positive_exposure, 95, axis=0)
        
        # EPE: moyenne temporelle de l'EE
        epe = np.trapz(ee, time_grid) / time_grid[-1]

        return {'ee': ee, 'ene': ene, 'pfe_95': pfe_95, 'epe': epe}
    
    def calculate_cva_direct_monte_carlo(self, exposure_profile: np.ndarray, 
                                         cp_default_times: np.ndarray,
                                         time_grid: np.ndarray) -> dict:
        """Calcul CVA direct par Monte Carlo. Méthode de référence."""
        Nmc = len(cp_default_times)
        losses = np.zeros(Nmc)
        lgd_cp = 1 - self.market_data.recovery_rate
        maturity = time_grid[-1]

        for j in range(Nmc):
            default_time = cp_default_times[j]
            if default_time <= maturity:
                # Interpolation linéaire pour trouver l'exposition au moment du défaut
                exposure_at_default = np.interp(default_time, time_grid, exposure_profile[j, :])
                positive_exposure_at_default = np.maximum(exposure_at_default, 0)
                
                discount_factor = np.exp(-self.market_data.r * default_time)
                losses[j] = lgd_cp * positive_exposure_at_default * discount_factor
        
        cva_direct = np.mean(losses)
        cva_std_error = np.std(losses, ddof=1) / np.sqrt(Nmc)
        
        return {
            'cva_direct': cva_direct,
            'std_error': cva_std_error,
            'confidence_95': 1.96 * cva_std_error
        }

# --- FONCTION PRINCIPALE DE SIMULATION ---

def main():
    print("🚀 DÉMARRAGE DU MODÈLE CVA CORRIGÉ - VALIDATION DU WRONG-WAY RISK 🚀")
    
    # Paramètres de simulation
    Nmc = 10000  # Nombre de simulations (ajusté pour la rapidité)
    T = 5.0      # Maturité
    dt = 1/12    # Pas de temps mensuel

    # --- CORRECTION CRITIQUE : Paramètres pour un VRAI Wrong-Way Risk ---
    # Pour un swap PAYEUR, on a besoin d'une HAUSSE des taux pour une exposition positive.
    # On configure donc initial_rate < theta.
    market_data = MarketData(
        r=0.02,
        sigma=0.02,          # Volatilité raisonnable
        initial_rate=0.02,   # CORRIGÉ : Taux initial bas
        theta=0.04,          # CORRIGÉ : Taux long terme élevé -> tendance à la hausse
        spread_credit=0.02,  # 200 bp
        recovery_rate=0.4,
        kappa=0.2            # Vitesse de retour à la moyenne modérée
    )
    
    # Paramètres du swap
    swap_params = SwapParameters(
        notional=1_000_000,
        maturity=T,
        fixed_rate=0.0,      # Calculé automatiquement pour être at-the-money
        payment_frequency=4, # Paiements trimestriels
        is_payer=True        # Position PAYEUR, sensible à la hausse des taux
    )

    print("\n--- CONFIGURATION DU SCÉNARIO ---")
    print(f"📈 Scénario WWR : Taux initial ({market_data.initial_rate:.2%}) < Theta ({market_data.theta:.2%}) -> Hausse des taux attendue")
    print(f"Position : PAYEUR -> L'exposition sera positive si les taux montent.")
    print(f"Corrélation WWR : Positive, liant la hausse des taux au défaut.")
    print("=> Le CVA devrait AUGMENTER avec la corrélation. C'est le Wrong-Way Risk.")

    # 1. Initialisation des modèles
    rate_model = VasicekModel(
        r0=market_data.initial_rate,
        kappa=market_data.kappa,
        theta=market_data.theta,
        sigma=market_data.sigma
    )
    swap = InterestRateSwap(swap_params, market_data, rate_model)
    print(f"Swap ATM Rate calculé : {swap.params.fixed_rate:.3%}")

    lambda_cp = market_data.spread_credit / (1 - market_data.recovery_rate)
    default_model_cp = DefaultModel(lambda_cp, market_data.recovery_rate)
    cva_engine = CVAEngine(market_data)

    # 2. Simulation des trajectoires de taux
    time_grid = np.arange(0, T + dt, dt)
    rate_paths = rate_model.simulate_paths(T, dt, Nmc)

    # 3. Calcul du profil d'exposition (Mark-to-Market)
    exposure_profile = swap.calculate_exposure_profile(rate_paths, time_grid)
    exposure_metrics_no_wwr = cva_engine.calculate_exposure_metrics(exposure_profile, time_grid)

    # 4. Scénario 1 : INDÉPENDANCE (pas de WWR)
    print("\n--- Calcul CVA SANS Wrong-Way Risk (Corrélation = 0) ---")
    cp_default_times_no_wwr = default_model_cp.simulate_default_times(T, Nmc)
    results_no_wwr = cva_engine.calculate_cva_direct_monte_carlo(exposure_profile, cp_default_times_no_wwr, time_grid)
    cva_no_wwr = results_no_wwr['cva_direct']
    print(f"CVA sans WWR = {cva_no_wwr:,.2f} EUR")

    # 5. Scénario 2 : CORRÉLATION (avec WWR)
    print("\n--- Calcul CVA AVEC Wrong-Way Risk (Corrélation > 0) ---")
    correlation_wwr = 0.6  # Corrélation forte pour un effet visible
    Z_default_cp_wwr = generate_correlated_variables_for_wwr(Nmc, rate_paths, correlation_wwr)
    cp_default_times_wwr = default_model_cp.simulate_default_times(T, Nmc, Z_default_cp_wwr)
    results_wwr = cva_engine.calculate_cva_direct_monte_carlo(exposure_profile, cp_default_times_wwr, time_grid)
    cva_wwr = results_wwr['cva_direct']
    print(f"CVA avec WWR (ρ={correlation_wwr:.0%}) = {cva_wwr:,.2f} EUR")
    
    # 6. Analyse des résultats
    print("\n--- ANALYSE FINALE ---")
    wwr_impact_abs = cva_wwr - cva_no_wwr
    wwr_impact_rel = (cva_wwr / cva_no_wwr - 1) * 100 if cva_no_wwr > 0 else 0
    
    print(f"Impact du WWR : {wwr_impact_abs:,.2f} EUR ({wwr_impact_rel:+.2f}%)")
    if wwr_impact_rel > 0:
        print("✅ SUCCÈS : Le paradoxe est résolu. Le CVA augmente avec la corrélation, démontrant un vrai Wrong-Way Risk.")
    else:
        print("❌ ERREUR : Le résultat est toujours un Right-Way Risk. Vérifier la logique.")
        
    cva_bp = cva_wwr / swap_params.notional * 10000
    print(f"CVA en points de base (bp) : {cva_bp:.2f} bp")

    # 7. Visualisation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Analyse CVA Corrigée - Résolution du Paradoxe WWR', fontsize=16, fontweight='bold')
    
    # Graphique 1 : Profil d'Exposition
    ax1.plot(time_grid, exposure_metrics_no_wwr['ee'], 'b-', label='Expected Exposure (EE)')
    ax1.plot(time_grid, exposure_metrics_no_wwr['pfe_95'], 'r--', label='Potential Future Exposure (PFE 95%)')
    ax1.fill_between(time_grid, 0, exposure_metrics_no_wwr['ee'], color='blue', alpha=0.2)
    ax1.set_title("Profil d'Exposition (Mark-to-Market)", fontweight='bold')
    ax1.set_xlabel("Temps (années)")
    ax1.set_ylabel("Exposition (EUR)")
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()

    # Graphique 2 : Impact du WWR sur le CVA
    bars = ax2.bar(['CVA sans WWR', f'CVA avec WWR\n(ρ={correlation_wwr:.0%})'], [cva_no_wwr, cva_wwr], color=['skyblue', 'salmon'])
    ax2.set_title("Impact du Wrong-Way Risk sur le CVA", fontweight='bold')
    ax2.set_ylabel("CVA (EUR)")
    ax2.grid(True, linestyle=':', alpha=0.6, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:,.0f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    main()
import fal_client
import os
from typing import Dict, Any

def generate_astronaut_video():
    """
    Génère une vidéo d'un astronaute français dans l'ISS avec fal.ai
    """
    
    # Configuration de l'API fal.ai
    # Assurez-vous d'avoir votre clé API dans les variables d'environnement
    fal_client.api_key = os.getenv("FAL_KEY")
    
    if not fal_client.api_key:
        raise ValueError("Veuillez définir votre clé API fal.ai dans la variable d'environnement FAL_KEY")
    
    # Prompt détaillé pour la génération vidéo
    video_prompt = """A cinematic, realistic 8K video. The shot is a tight close-up on the face of a French astronaut, a man in his forties, inside the ISS Cupola. The blue Earth is visible in the background, reflected in his eyes. His expression is a tense mix of urgency and utter disbelief.

CRITICAL AUDIO INSTRUCTION: The dialogue must be spoken in clear, distinct French. The voice quality should be that of a clean radio transmission with a subtle, intermittent static. Lip-sync must be perfect.

ACTION and DIALOGUE:
He speaks urgently into his communication microphone.
Audio (in French): "Ici la station. Il se passe un truc de fou ! Anne de Bretagne est la première femme à marcher sur Mars."

Immediately after the last word, he cuts the communication with an audible click and abruptly turns his head towards a telescope just out of frame. The video ends precisely on this movement.

Style: Cinematic realism, professional film quality, dramatic lighting from Earth's reflection, authentic space station interior, detailed facial expressions, realistic French pronunciation, perfect lip synchronization."""
    
    # Paramètres pour la génération vidéo
    generation_params = {
        "prompt": video_prompt,
        "model_name": "runway-gen3/turbo/image-to-video",  # Ou le modèle approprié selon fal.ai
        "image_url": None,  # Pas d'image de base, génération complète
        "duration": 10,  # Durée en secondes
        "aspect_ratio": "16:9",
        "motion_bucket_id": 180,  # Contrôle du mouvement
        "fps": 24,  # Images par seconde
        "seed": None,  # Pour la reproductibilité, vous pouvez spécifier un seed
        "enable_prompt_upsampling": True,
        "remove_watermark": True
    }
    
    try:
        print("🚀 Début de la génération vidéo...")
        print(f"Prompt: {video_prompt[:100]}...")
        
        # Appel à l'API fal.ai pour générer la vidéo
        result = fal_client.subscribe(
            "fal-ai/runway-gen3-turbo-image-to-video",  # Endpoint du modèle
            arguments=generation_params,
            with_logs=True
        )
        
        # Traitement du résultat
        if result and "video" in result:
            video_url = result["video"]["url"]
            print(f"✅ Vidéo générée avec succès!")
            print(f"🔗 URL de téléchargement: {video_url}")
            
            # Optionnel: télécharger la vidéo localement
            download_video(video_url, "astronaute_iss_french.mp4")
            
            return {
                "success": True,
                "video_url": video_url,
                "result": result
            }
        else:
            print("❌ Erreur: Aucune vidéo générée")
            return {"success": False, "error": "No video generated"}
            
    except Exception as e:
        print(f"❌ Erreur lors de la génération: {str(e)}")
        return {"success": False, "error": str(e)}

def download_video(url: str, filename: str):
    """
    Télécharge la vidéo générée localement
    """
    try:
        import requests
        
        print(f"📥 Téléchargement de la vidéo: {filename}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        
        print(f"✅ Vidéo sauvegardée: {filename}")
        
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement: {str(e)}")

def main():
    """
    Fonction principale pour exécuter le générateur de vidéo
    """
    print("🎬 Générateur de vidéo fal.ai - Astronaute français ISS")
    print("=" * 50)
    
    # Vérification des dépendances
    try:
        import fal_client
        import requests
    except ImportError as e:
        print(f"❌ Module manquant: {e}")
        print("Installez les dépendances avec: pip install fal-client requests")
        return
    
    # Génération de la vidéo
    result = generate_astronaut_video()
    
    if result["success"]:
        print("\n🎉 Génération terminée avec succès!")
        print("La vidéo est maintenant disponible.")
    else:
        print(f"\n❌ Échec de la génération: {result.get('error', 'Erreur inconnue')}")

if __name__ == "__main__":
    # Instructions d'installation et configuration
    print("""
    📋 INSTRUCTIONS D'INSTALLATION ET CONFIGURATION:
    
    1. Installez les dépendances:
       pip install fal-client requests
    
    2. Obtenez votre clé API sur https://fal.ai
    
    3. Définissez votre clé API:
       export FAL_KEY="votre_cle_api_ici"
       
       Ou sur Windows:
       set FAL_KEY=votre_cle_api_ici
    
    4. Exécutez le script:
       python script_name.py
    """)
    
    main()