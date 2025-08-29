import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional
import scipy.stats as stats

@dataclass
class MarketData:
    """Données de marché pour le pricing"""
    r: float  # taux sans risque
    sigma: float  # volatilité
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

def generate_correlated_variables_for_wwr(Nmc: int, rate_paths: np.ndarray, 
                                        correlation_wwr: float) -> np.ndarray:
    """
    Génère des variables corrélées pour le Wrong-Way Risk.
    
    Pour un swap PAYER, le WWR survient quand une hausse des taux
    (qui augmente l'exposition) est corrélée à une hausse du risque de défaut.
    
    Approche: Corrélation avec les taux finaux (représentent l'impact cumulé)
    """
    # Facteur systémique basé sur les taux finaux
    rate_systemic = rate_paths[:, -1]  # Taux à maturité
    rate_normalized = (rate_systemic - np.mean(rate_systemic)) / np.std(rate_systemic)
    
    # Variable indépendante pour le défaut
    Z_independent = np.random.normal(0, 1, Nmc)
    
    # Corrélation via formule de Cholesky
    Z_correlated = (correlation_wwr * rate_normalized + 
                   np.sqrt(1 - correlation_wwr**2) * Z_independent)
    
    return Z_correlated

class VasicekModel:
    """
    Modèle de Vasicek: dr = κ(θ - r)dt + σdW
    
    Si r0 < θ, les taux montent en moyenne (courbe ascendante)
    Si r0 > θ, les taux baissent en moyenne (courbe inversée)
    """
    
    def __init__(self, r0: float, kappa: float, theta: float, sigma: float):
        self.r0 = r0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        
    def simulate_paths(self, T: float, dt: float, Nmc: int) -> np.ndarray:
        """Simule les trajectoires de taux avec dérive correcte"""
        n_steps = int(round(T / dt))
        rate_paths = np.zeros((Nmc, n_steps + 1))
        rate_paths[:, 0] = self.r0
        
        dW = np.random.normal(0, np.sqrt(dt), (Nmc, n_steps))
        
        for i in range(n_steps):
            # Dérive de retour à la moyenne: κ(θ - r)
            drift = self.kappa * (self.theta - rate_paths[:, i]) * dt
            diffusion = self.sigma * dW[:, i]
            rate_paths[:, i + 1] = rate_paths[:, i] + drift + diffusion
            
            # Éviter les taux extrêmement négatifs
            rate_paths[:, i + 1] = np.maximum(rate_paths[:, i + 1], -0.02)
            
        return rate_paths
    
    def zero_coupon_bond(self, tau: float, r_current: np.ndarray) -> np.ndarray:
        """
        Prix du zéro-coupon Vasicek: P(t,T) = A(τ) * exp(-B(τ) * r(t))
        où τ = T - t est la maturité résiduelle
        """
        if self.kappa == 0:
            B = tau
            A = np.exp(-self.theta * tau + (self.sigma**2 * tau**3) / 6)
        else:
            B = (1 - np.exp(-self.kappa * tau)) / self.kappa
            term1 = (self.theta - self.sigma**2 / (2 * self.kappa**2)) * (B - tau)
            term2 = (self.sigma**2 * B**2) / (4 * self.kappa)
            A = np.exp(term1 - term2)
        
        return A * np.exp(-B * r_current)

class InterestRateSwap:
    """
    Swap de taux d'intérêt avec valorisation rigoureuse
    
    Re-valorisation complète du swap à chaque pas de temps
    en utilisant les formules analytiques Vasicek
    """
    
    def __init__(self, params: SwapParameters, market_data: MarketData, 
                 rate_model: VasicekModel):
        self.params = params
        self.market_data = market_data
        self.rate_model = rate_model
        self.payment_dates = self._generate_payment_dates()
        
        # Calcul du taux ATM si nécessaire
        if self.params.fixed_rate == 0.0:
            self.params.fixed_rate = self._calculate_atm_rate()
        
    def _generate_payment_dates(self) -> np.ndarray:
        """Génère les dates de paiement"""
        dt = 1.0 / self.params.payment_frequency
        return np.arange(dt, self.params.maturity + dt, dt)
    
    def _calculate_atm_rate(self) -> float:
        """
        Calcul rigoureux du taux swap at-the-money
        
        Formule: R_ATM = (1 - P(0,Tn)) / Σ(Δt * P(0,Ti))
        """
        dt = 1.0 / self.params.payment_frequency
        
        # Prix des zéro-coupons pour chaque date de paiement
        zcb_prices = []
        for date in self.payment_dates:
            price = self.rate_model.zero_coupon_bond(date, np.array([self.market_data.initial_rate]))[0]
            zcb_prices.append(price)
        
        zcb_prices = np.array(zcb_prices)
        annuity = np.sum(dt * zcb_prices)
        
        if annuity > 0:
            return (1 - zcb_prices[-1]) / annuity
        else:
            return self.market_data.initial_rate
    
    def calculate_npv_paths(self, rate_paths: np.ndarray, time_grid: np.ndarray) -> np.ndarray:
        """
        Calcul de la NPV par re-valorisation complète
        
        Théorie: 
        - NPV(t) = PV_flottant(t) - PV_fixe(t) pour un PAYER
        - PV_fixe(t) = K * N * Σ P(t, Ti, r(t)) * Δt
        - PV_flottant(t) = N * (1 - P(t, Tn, r(t)))
        """
        Nmc, n_steps = rate_paths.shape
        npv_paths = np.zeros_like(rate_paths)
        dt_payment = 1.0 / self.params.payment_frequency

        # La NPV à maturité est nulle
        npv_paths[:, -1] = 0.0

        for i in range(n_steps - 1):
            t = time_grid[i]
            r_t = rate_paths[:, i]
            
            # Dates de paiement futures à partir de t
            future_payment_dates = self.payment_dates[self.payment_dates > t]
            
            if len(future_payment_dates) == 0:
                npv_paths[:, i] = 0.0
                continue
            
            # Valeur de la jambe fixe
            pv_fixed_leg = np.zeros(Nmc)
            for payment_date in future_payment_dates:
                tau = payment_date - t
                if tau > 0:
                    zc_prices = self.rate_model.zero_coupon_bond(tau, r_t)
                    pv_fixed_leg += zc_prices * dt_payment
            
            pv_fixed_leg *= self.params.fixed_rate * self.params.notional

            # Valeur de la jambe flottante
            maturity_tau = self.params.maturity - t
            if maturity_tau > 0:
                zc_maturity = self.rate_model.zero_coupon_bond(maturity_tau, r_t)
                pv_float_leg = self.params.notional * (1 - zc_maturity)
            else:
                pv_float_leg = np.zeros(Nmc)

            # NPV selon la position
            if self.params.is_payer:
                npv_paths[:, i] = pv_float_leg - pv_fixed_leg
            else:
                npv_paths[:, i] = pv_fixed_leg - pv_float_leg
                
        return npv_paths

class DefaultModel:
    """Modèle de défaut avec intensité constante"""
    
    def __init__(self, lambda_default: float, recovery_rate: float):
        self.lambda_default = lambda_default
        self.recovery_rate = recovery_rate
        
    def simulate_default_times(self, T: float, Nmc: int, 
                             Z_latent: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Simule les temps de défaut via copule gaussienne
        
        Logique:
        - Z_latent élevé (forte exposition, mauvaise situation) 
        - → U = cdf(-Z_latent) faible (proche de 0)
        - → temps de défaut = -log(U)/lambda court (défaut précoce)
        """
        if Z_latent is None:
            Z_latent = np.random.normal(0, 1, Nmc)
        
        # Un Z_latent élevé mène à un défaut précoce (temps court)
        U = stats.norm.cdf(-Z_latent)
        U = np.clip(U, 1e-10, 1-1e-10)  # Évite log(0)
        
        return -np.log(U) / self.lambda_default

class CVAEngine:
    """
    Moteur CVA avec actualisation stochastique cohérente
    
    Utilisation des facteurs d'actualisation stochastiques
    au lieu d'un taux constant
    """
    
    def __init__(self, market_data: MarketData):
        self.market_data = market_data
        
    def calculate_exposure_metrics(self, npv_paths: np.ndarray, time_grid: np.ndarray) -> dict:
        """Calcule les métriques d'exposition standards"""
        positive_exposure = np.maximum(npv_paths, 0)
        
        ee = np.mean(positive_exposure, axis=0)  # Expected Exposure
        pfe_95 = np.percentile(positive_exposure, 95, axis=0)  # PFE 95%
        pfe_99 = np.percentile(positive_exposure, 99, axis=0)  # PFE 99%
        
        # Calcul de l'EPE
        epe = np.mean(ee)
        
        return {
            'ee': ee, 
            'pfe_95': pfe_95, 
            'pfe_99': pfe_99,
            'epe': epe,
            'max_pfe': np.max(pfe_95)
        }
    
    def calculate_stochastic_discount_factors(self, rate_paths: np.ndarray, 
                                            time_grid: np.ndarray) -> np.ndarray:
        """
        Calcul des facteurs d'actualisation stochastiques
        
        D(0,t) = exp(-∫₀ᵗ r(s)ds)
        """
        # S'assurer que les dimensions correspondent
        n_rate_steps = rate_paths.shape[1]
        n_time_steps = len(time_grid)
        
        if n_rate_steps == n_time_steps:
            dt = time_grid[1] - time_grid[0] if len(time_grid) > 1 else 0.0
            rates_for_integration = rate_paths
        elif n_rate_steps == n_time_steps - 1:
            dt = time_grid[1] - time_grid[0] if len(time_grid) > 1 else 0.0
            rates_for_integration = rate_paths
        else:
            # Ajuster en prenant les n_time_steps premiers éléments
            dt = time_grid[1] - time_grid[0] if len(time_grid) > 1 else 0.0
            rates_for_integration = rate_paths[:, :min(n_rate_steps, n_time_steps)]
        
        # Intégration trapézoïdale des taux
        integrated_rates = np.cumsum(rates_for_integration * dt, axis=1)
        
        # Facteurs d'actualisation stochastiques
        return np.exp(-integrated_rates)
    
    def calculate_cva_direct(self, npv_paths: np.ndarray, 
                           default_times: np.ndarray,
                           time_grid: np.ndarray,
                           rate_paths: np.ndarray) -> dict:
        """
        CVA direct avec actualisation stochastique path-dependent
        
        CVA = LGD * E[1_{τ≤T} * D(0,τ) * EE(τ)]
        où D(0,τ) = exp(-∫₀^τ r(s)ds) est stochastique
        """
        Nmc = len(default_times)
        losses = np.zeros(Nmc)
        lgd = 1 - self.market_data.recovery_rate
        
        # Calcul des facteurs d'actualisation stochastiques
        discount_factors = self.calculate_stochastic_discount_factors(rate_paths, time_grid)
        
        for j in range(Nmc):
            default_time = default_times[j]
            
            if default_time <= time_grid[-1]:
                # Trouve l'indice temporel du défaut
                default_idx = np.searchsorted(time_grid, default_time, side='right') - 1
                default_idx = np.clip(default_idx, 0, len(time_grid) - 1)
                
                # Exposition positive au moment du défaut
                exposure = max(0, npv_paths[j, default_idx])
                
                # Facteur d'actualisation stochastique pour cette trajectoire
                discount = discount_factors[j, default_idx]
                
                # Perte actualisée
                losses[j] = lgd * exposure * discount
        
        cva_direct = np.mean(losses)
        cva_std_error = np.std(losses, ddof=1) / np.sqrt(Nmc)
        
        return {
            'cva_direct': cva_direct,
            'std_error': cva_std_error,
            'confidence_95': 1.96 * cva_std_error
        }

def main():
    """
    Fonction principale - Modèle CVA avec Wrong-Way Risk
    
    Logique théorique:
    1. Configuration: r₀ < θ → taux montent en moyenne (courbe ascendante)
    2. Swap PAYER: exposition positive si taux > taux fixe
    3. Corrélation: taux élevés → Z_latent élevé → défaut précoce (WWR)
    4. Résultat attendu: CVA avec WWR > CVA sans WWR
    """
    
    print("=== MODÈLE CVA - WRONG-WAY RISK ===")
    
    # Paramètres de simulation
    Nmc = 20000  # Nombre de simulations Monte Carlo
    T = 5.0      # Maturité
    dt = 1/24    # Pas de temps
    
    # Configuration pour Wrong-Way Risk
    # Courbe ascendante: initial_rate < theta pour que les taux montent en moyenne
    market_data = MarketData(
        r=0.02,              # Taux de référence
        sigma=0.02,          # Volatilité modérée  
        initial_rate=0.02,   # 2.0% < theta
        theta=0.04,          # 4.0% > initial_rate  
        spread_credit=0.015, # 150 bp pour CVA réaliste
        recovery_rate=0.4,
        kappa=0.2            # Vitesse de retour modérée
    )
    
    # Paramètres du swap PAYER
    swap_params = SwapParameters(
        notional=1_000_000,
        maturity=T,
        fixed_rate=0.0,      # Calculé automatiquement
        payment_frequency=4, # Trimestriel
        is_payer=True        # Position PAYER sensible à la hausse des taux
    )
    
    print(f"Configuration:")
    print(f"- Courbe taux: r₀={market_data.initial_rate:.1%} < θ={market_data.theta:.1%}")
    print(f"- Direction attendue: Hausse des taux")
    print(f"- Position swap: PAYER (exposition positive si taux montent)")
    
    # 1. Initialisation des modèles
    print("\n1. Initialisation du modèle Vasicek...")
    rate_model = VasicekModel(
        market_data.initial_rate, market_data.kappa, 
        market_data.theta, market_data.sigma
    )
    
    # Vérification de la courbe des taux
    zcb_1y = rate_model.zero_coupon_bond(1.0, np.array([market_data.initial_rate]))[0]
    zcb_5y = rate_model.zero_coupon_bond(5.0, np.array([market_data.initial_rate]))[0]
    print(f"   P(0,1Y) = {zcb_1y:.4f}")
    print(f"   P(0,5Y) = {zcb_5y:.4f}")
    print(f"   Courbe: {'Ascendante' if zcb_1y > zcb_5y else 'Descendante'}")
    
    # 2. Calcul du taux swap ATM
    print("\n2. Calcul du swap ATM...")
    swap = InterestRateSwap(swap_params, market_data, rate_model)
    print(f"   Taux initial: {market_data.initial_rate:.3%}")
    print(f"   Taux ATM: {swap.params.fixed_rate:.3%}")
    
    # 3. Simulation des trajectoires
    print("\n3. Simulation des trajectoires de taux...")
    n_steps = int(round(T / dt))
    time_grid = np.linspace(0, T, n_steps + 1)
    rate_paths = rate_model.simulate_paths(T, dt, Nmc)
    
    print(f"   Taux final moyen: {np.mean(rate_paths[:, -1]):.3%}")
    print(f"   Convergence vers θ: {abs(np.mean(rate_paths[:, -1]) - market_data.theta) < 0.01}")
    
    # 4. Calcul de l'exposition par re-valorisation
    print("\n4. Calcul de l'exposition par re-valorisation...")
    npv_paths = swap.calculate_npv_paths(rate_paths, time_grid)
    
    print(f"   NPV initiale moyenne: {np.mean(npv_paths[:, 0]):,.0f} EUR")
    print(f"   Max exposition: {np.max(npv_paths):,.0f} EUR")
    print(f"   At-the-money: {abs(np.mean(npv_paths[:, 0])) < 1000}")
    
    # 5. Modèles de défaut
    print("\n5. Simulation des défauts...")
    lambda_cp = market_data.spread_credit / (1 - market_data.recovery_rate)
    default_model = DefaultModel(lambda_cp, market_data.recovery_rate)
    
    print(f"   Intensité de défaut: {lambda_cp:.4f}")
    print(f"   Probabilité de survie 5Y: {np.exp(-lambda_cp * T):.2%}")
    
    # 6. CVA Engine
    cva_engine = CVAEngine(market_data)
    exposure_metrics = cva_engine.calculate_exposure_metrics(npv_paths, time_grid)
    
    # 7. Scénario 1: Sans WWR (indépendance)
    print("\n7. CVA sans Wrong-Way Risk...")
    default_times_indep = default_model.simulate_default_times(T, Nmc)
    results_no_wwr = cva_engine.calculate_cva_direct(
        npv_paths, default_times_indep, time_grid, rate_paths
    )
    cva_no_wwr = results_no_wwr['cva_direct']
    print(f"   CVA sans WWR: {cva_no_wwr:,.0f} EUR")
    
    # 8. Scénario 2: Avec WWR (corrélation)
    print("\n8. CVA avec Wrong-Way Risk...")
    correlation_wwr = -0.5
    
    Z_correlated = generate_correlated_variables_for_wwr(Nmc, rate_paths, correlation_wwr)
    default_times_wwr = default_model.simulate_default_times(T, Nmc, Z_correlated)
    results_wwr = cva_engine.calculate_cva_direct(
        npv_paths, default_times_wwr, time_grid, rate_paths
    )
    cva_wwr = results_wwr['cva_direct']
    print(f"   CVA avec WWR (ρ={correlation_wwr:.0%}): {cva_wwr:,.0f} EUR")
    
    # 9. Analyse de l'impact WWR
    print("\n9. Analyse des résultats:")
    impact_abs = cva_wwr - cva_no_wwr
    impact_rel = (cva_wwr / cva_no_wwr - 1) * 100 if cva_no_wwr > 0 else 0
    
    print(f"   Impact WWR: {impact_abs:,.0f} EUR ({impact_rel:+.1f}%)")
    print(f"   CVA en bp: {cva_wwr/swap_params.notional*10000:.1f}")
    print(f"   Direction: {'Wrong-Way' if impact_rel > 0 else 'Right-Way'}")
    print(f"   EPE: {exposure_metrics['epe']:,.0f} EUR")
    
    # 10. Validation empirique de la corrélation
    rate_factor = rate_paths[:, -1]  # Taux finaux utilisés pour la corrélation
    empirical_corr = np.corrcoef(rate_factor, Z_correlated)[0, 1]
    print(f"   Corrélation empirique: {empirical_corr:.3f}")
    
    # 11. Visualisation des résultats
    print("\n11. Génération des graphiques...")
    create_validation_plots(
        time_grid, rate_paths, npv_paths, exposure_metrics,
        cva_no_wwr, cva_wwr, market_data, empirical_corr, swap.payment_dates
    )
    
    # Résumé final
    print("\n" + "="*60)
    print("RÉSUMÉ - WRONG-WAY RISK")
    print("="*60)
    
    cva_wwr_bp = cva_wwr/swap_params.notional*10000
    cva_no_wwr_bp = cva_no_wwr/swap_params.notional*10000
    
    print(f"Configuration théorique:")
    print(f"- Courbe des taux: r₀={market_data.initial_rate:.1%} < θ={market_data.theta:.1%}")
    print(f"- Position swap: PAYER (sensible à la hausse)")
    print(f"- Corrélation WWR: ρ={correlation_wwr:.1%}")
    print("-" * 60)
    
    print(f"Résultats:")
    print(f"CVA sans WWR (bp): {cva_no_wwr_bp:.1f}")
    print(f"CVA avec WWR (bp): {cva_wwr_bp:.1f}")
    print(f"Impact WWR (bp): {(cva_wwr_bp - cva_no_wwr_bp):.1f}")
    print(f"IC Monte Carlo: ±{results_wwr['confidence_95']:.0f} EUR")
    print("-" * 60)
    
    print(f"Validation:")
    print(f"WWR direction: {'Wrong-Way' if impact_rel > 0 else 'Right-Way'}")
    print(f"CVA réaliste: {cva_wwr_bp:.1f} bp")
    print(f"Impact significatif: {impact_rel:.1f}%")
    
    success = (impact_rel > 0 and cva_wwr_bp > 5 and abs(impact_rel) > 10)
    print("="*60)
    print(f"{'Wrong-Way Risk validé' if success else 'Vérifications requises'}")
    print("="*60)

def create_validation_plots(time_grid, rate_paths, npv_paths, exposure_metrics,
                          cva_no_wwr, cva_wwr, market_data, empirical_corr, payment_dates):
    """Graphiques de validation du modèle"""
    
    # S'assurer que les dimensions correspondent
    n_time = min(len(time_grid), rate_paths.shape[1])
    time_plot = time_grid[:n_time]
    
    # 1. Trajectoires de taux
    plt.figure(figsize=(10, 6))
    sample_size = min(100, rate_paths.shape[0])
    
    for i in range(sample_size):
        plt.plot(time_plot, rate_paths[i, :n_time], alpha=0.1, color='steelblue')
    plt.plot(time_plot, np.mean(rate_paths[:, :n_time], axis=0), 'darkred', linewidth=3,
             label=f'Moyenne (→{np.mean(rate_paths[:, -1]):.1%})')
    plt.axhline(y=market_data.theta, color='green', linestyle='--', linewidth=2,
               label=f'θ = {market_data.theta:.1%}')
    plt.axhline(y=market_data.initial_rate, color='orange', linestyle=':', linewidth=2,
               label=f'r₀ = {market_data.initial_rate:.1%}')
    plt.title('Trajectoires Vasicek', fontweight='bold', fontsize=14)
    plt.xlabel('Temps (années)')
    plt.ylabel('Taux')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 2. Profil d'exposition
    plt.figure(figsize=(10, 6))
    ee = exposure_metrics['ee']
    pfe_95 = exposure_metrics['pfe_95']
    
    time_ee = time_plot[:len(ee)]
    
    plt.plot(time_ee, ee[:len(time_ee)], 'blue', linewidth=3, label='Expected Exposure')
    plt.plot(time_ee, pfe_95[:len(time_ee)], 'red', linewidth=2, label='PFE 95%')
    plt.fill_between(time_ee, 0, ee[:len(time_ee)], alpha=0.3, color='blue')
    plt.title('Profil d\'Exposition', fontweight='bold', fontsize=14)
    plt.xlabel('Temps (années)')
    plt.ylabel('Exposition (EUR)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 3. Comparaison CVA
    plt.figure(figsize=(8, 6))
    cva_values = [cva_no_wwr, cva_wwr]
    labels = ['Sans WWR', f'Avec WWR\n(ρ={empirical_corr:.2f})']
    colors = ['blue', 'red']
    
    bars = plt.bar(labels, cva_values, color=colors, alpha=0.8, edgecolor='black')
    plt.title('Comparaison CVA', fontweight='bold', fontsize=14)
    plt.ylabel('CVA (EUR)')
    plt.grid(True, alpha=0.3, axis='y')
    
    for bar, value in zip(bars, cva_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                f'{value:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    if cva_no_wwr > 0:
        impact_pct = (cva_wwr / cva_no_wwr - 1) * 100
        color = 'green' if impact_pct > 0 else 'red'
        plt.text(0.5, max(cva_values)*0.7, f'Impact WWR:\n{impact_pct:+.1f}%',
                ha='center', va='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.2),
                transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.show()
    
    # 4. Distribution NPV finale
    plt.figure(figsize=(10, 6))
    npv_final = npv_paths[:, -1]
    plt.hist(npv_final, bins=50, alpha=0.7, density=True, color='lightblue', edgecolor='black')
    
    mean_npv = np.mean(npv_final)
    plt.axvline(mean_npv, color='red', linestyle='--', linewidth=2,
               label=f'Moyenne: {mean_npv:,.0f}')
    plt.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    plt.title('Distribution NPV Finale', fontweight='bold', fontsize=14)
    plt.xlabel('NPV finale (EUR)')
    plt.ylabel('Densité')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()