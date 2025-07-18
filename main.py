import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class MarketData:
    """Données de marché pour le pricing"""
    r: float  # taux sans risque
    sigma: float  # volatilité du taux
    initial_rate: float  # taux initial
    spread_credit: float  # spread de crédit
    recovery_rate: float  # taux de récupération
    
@dataclass
class SwapParameters:
    """Paramètres du swap"""
    notional: float
    maturity: float  # en années
    fixed_rate: float
    payment_frequency: int  # paiements par an
    is_payer: bool  # True si on paie le taux fixe
    
class InterestRateModel(ABC):
    """Classe abstraite pour les modèles de taux"""
    
    @abstractmethod
    def simulate_paths(self, T: float, dt: float, Nmc: int) -> np.ndarray:
        """Simule les trajectoires de taux"""
        pass

class GeometricBrownianMotion(InterestRateModel):
    """Modèle de Mouvement Brownien Géométrique pour les taux"""
    
    def __init__(self, r0: float, mu: float, sigma: float):
        self.r0 = r0  # taux initial
        self.mu = mu  # drift
        self.sigma = sigma  # volatilité
        
    def simulate_paths(self, T: float, dt: float, Nmc: int) -> np.ndarray:
        """
        Simule les trajectoires de taux avec MBG
        Returns: array de forme (Nmc, n_steps)
        """
        n_steps = int(T / dt)
        paths = np.zeros((Nmc, n_steps + 1))
        paths[:, 0] = self.r0
        
        # Génération des innovations aléatoires
        dW = np.random.normal(0, np.sqrt(dt), (Nmc, n_steps))
        
        for i in range(n_steps):
            paths[:, i + 1] = paths[:, i] * np.exp(
                (self.mu - 0.5 * self.sigma**2) * dt + self.sigma * dW[:, i]
            )
            
        return paths

class DefaultModel:
    """Modèle de défaut avec intensité constante"""
    
    def __init__(self, lambda_default: float, recovery_rate: float):
        self.lambda_default = lambda_default  # intensité de défaut
        self.recovery_rate = recovery_rate
        
    def simulate_default_times(self, T: float, Nmc: int) -> np.ndarray:
        """Simule les temps de défaut exponentiels"""
        return np.random.exponential(1 / self.lambda_default, Nmc)
    
    def survival_probability(self, t: float) -> float:
        """Probabilité de survie à l'instant t"""
        return np.exp(-self.lambda_default * t)

class GaussianCopula:
    """Copule Gaussienne pour la corrélation de défaut"""
    
    def __init__(self, correlation_matrix: np.ndarray):
        self.correlation_matrix = correlation_matrix
        self.n_entities = correlation_matrix.shape[0]
        
    def simulate_correlated_defaults(self, T: float, Nmc: int, 
                                   default_intensities: List[float]) -> np.ndarray:
        """
        Simule des temps de défaut corrélés via copule Gaussienne
        Returns: array de forme (Nmc, n_entities)
        """
        # Génération des variables gaussiennes corrélées
        Z = np.random.multivariate_normal(
            np.zeros(self.n_entities), 
            self.correlation_matrix, 
            Nmc
        )
        
        # Transformation en variables uniformes
        U = stats.norm.cdf(Z)
        
        # Transformation en temps de défaut
        default_times = np.zeros((Nmc, self.n_entities))
        for i, lambda_i in enumerate(default_intensities):
            default_times[:, i] = -np.log(1 - U[:, i]) / lambda_i
            
        return default_times

class InterestRateSwap:
    """Classe pour le pricing d'un Interest Rate Swap"""
    
    def __init__(self, swap_params: SwapParameters):
        self.params = swap_params
        self.payment_dates = self._generate_payment_dates()
        
    def _generate_payment_dates(self) -> np.ndarray:
        """Génère les dates de paiement"""
        dt_payment = 1.0 / self.params.payment_frequency
        return np.arange(dt_payment, self.params.maturity + dt_payment, dt_payment)
    
    def calculate_npv(self, rate_paths: np.ndarray, r: float, 
                     time_grid: np.ndarray) -> np.ndarray:
        """
        Calcule la NPV du swap pour chaque trajectoire
        Returns: array de forme (Nmc, n_payment_dates)
        """
        Nmc = rate_paths.shape[0]
        n_payments = len(self.payment_dates)
        npv_matrix = np.zeros((Nmc, n_payments))
        
        for i, payment_date in enumerate(self.payment_dates):
            # Trouver l'indice temporel le plus proche
            time_idx = np.argmin(np.abs(time_grid - payment_date))
            
            # Facteur d'actualisation
            discount_factor = np.exp(-r * payment_date)
            
            # Flux fixe
            fixed_flow = self.params.fixed_rate * self.params.notional / self.params.payment_frequency
            
            # Flux flottant (taux à la date de fixing)
            if time_idx < len(time_grid):
                floating_flow = rate_paths[:, time_idx] * self.params.notional / self.params.payment_frequency
            else:
                floating_flow = rate_paths[:, -1] * self.params.notional / self.params.payment_frequency
            
            # NPV selon la position (payer ou receveur du taux fixe)
            if self.params.is_payer:
                cash_flow = floating_flow - fixed_flow
            else:
                cash_flow = fixed_flow - floating_flow
                
            npv_matrix[:, i] = cash_flow * discount_factor
            
        return npv_matrix

class CVACalculator:
    """Calculateur de CVA"""
    
    def __init__(self, market_data: MarketData):
        self.market_data = market_data
        
    def calculate_exposure_metrics(self, npv_matrix: np.ndarray) -> dict:
        """
        Calcule les métriques d'exposition (EE, PFE, EPE)
        """
        # Expected Exposure (EE)
        ee = np.mean(np.maximum(npv_matrix, 0), axis=0)
        
        # Potential Future Exposure (PFE) au quantile 95%
        pfe_95 = np.percentile(np.maximum(npv_matrix, 0), 95, axis=0)
        
        # Expected Positive Exposure (EPE)
        epe = np.mean(ee)
        
        return {
            'ee': ee,
            'pfe_95': pfe_95,
            'epe': epe,
            'max_pfe': np.max(pfe_95)
        }
    
    def calculate_unilateral_cva(self, npv_matrix: np.ndarray, 
                               default_times: np.ndarray,
                               time_grid: np.ndarray) -> float:
        """Calcule le CVA unilatéral"""
        Nmc = npv_matrix.shape[0]
        cva_values = np.zeros(Nmc)
        
        for i in range(Nmc):
            default_time = default_times[i]
            
            # Trouver l'indice de défaut
            default_idx = np.searchsorted(time_grid, default_time)
            
            if default_idx < len(time_grid):
                # Exposition positive au moment du défaut
                exposure = np.maximum(npv_matrix[i, default_idx], 0)
                
                # CVA = LGD * EAD * DF
                lgd = 1 - self.market_data.recovery_rate
                discount_factor = np.exp(-self.market_data.r * default_time)
                cva_values[i] = lgd * exposure * discount_factor
                
        return np.mean(cva_values)
    
    def calculate_bilateral_cva(self, npv_matrix: np.ndarray,
                              counterparty_default_times: np.ndarray,
                              own_default_times: np.ndarray,
                              time_grid: np.ndarray) -> dict:
        """Calcule le CVA bilatéral (CVA - DVA)"""
        
        # CVA unilatéral
        cva = self.calculate_unilateral_cva(npv_matrix, counterparty_default_times, time_grid)
        
        # DVA (Debt Valuation Adjustment)
        dva_matrix = -npv_matrix  # Exposition négative pour nous
        dva = self.calculate_unilateral_cva(dva_matrix, own_default_times, time_grid)
        
        # CVA bilatéral
        bilateral_cva = cva - dva
        
        return {
            'cva': cva,
            'dva': dva,
            'bilateral_cva': bilateral_cva
        }

class CollateralModel:
    """Modèle de collatéral"""
    
    def __init__(self, threshold: float, minimum_transfer: float, 
                 haircut: float = 0.0):
        self.threshold = threshold
        self.minimum_transfer = minimum_transfer
        self.haircut = haircut
        
    def calculate_collateral_amount(self, exposure: float) -> float:
        """Calcule le montant de collatéral requis"""
        net_exposure = exposure - self.threshold
        
        if net_exposure > self.minimum_transfer:
            return net_exposure * (1 + self.haircut)
        else:
            return 0.0
    
    def adjust_exposure_for_collateral(self, npv_matrix: np.ndarray) -> np.ndarray:
        """Ajuste l'exposition pour le collatéral"""
        adjusted_npv = np.zeros_like(npv_matrix)
        
        for i in range(npv_matrix.shape[0]):
            for j in range(npv_matrix.shape[1]):
                exposure = npv_matrix[i, j]
                collateral = self.calculate_collateral_amount(exposure)
                adjusted_npv[i, j] = max(0, exposure - collateral)
                
        return adjusted_npv

# Exemple d'utilisation
def main():
    """Fonction principale pour démontrer l'utilisation"""
    
    # Paramètres de simulation
    Nmc = 10000
    T = 5.0  # 5 ans
    dt = 1/12  # pas mensuel
    
    # Données de marché
    market_data = MarketData(
        r=0.02,  # taux sans risque 2%
        sigma=0.15,  # volatilité 15%
        initial_rate=0.03,  # taux initial 3%
        spread_credit=0.01,  # spread 100bp
        recovery_rate=0.4  # récupération 40%
    )
    
    # Paramètres du swap
    swap_params = SwapParameters(
        notional=1000000,  # 1M
        maturity=T,
        fixed_rate=0.025,  # 2.5%
        payment_frequency=4,  # trimestriel
        is_payer=True
    )
    
    print("Simulation Monte Carlo en cours...")
    print(f"Nombre de simulations: {Nmc}")
    print(f"Maturité: {T} ans")
    print(f"Pas de temps: {dt}")
    
    # 1. Simulation des trajectoires de taux
    print("\n1. Simulation des trajectoires de taux...")
    rate_model = GeometricBrownianMotion(
        r0=market_data.initial_rate,
        mu=market_data.r,
        sigma=market_data.sigma
    )
    
    time_grid = np.arange(0, T + dt, dt)
    rate_paths = rate_model.simulate_paths(T, dt, Nmc)
    
    print(f"   Taux initial: {market_data.initial_rate:.2%}")
    print(f"   Taux final moyen: {np.mean(rate_paths[:, -1]):.2%}")
    print(f"   Volatilité réalisée: {np.std(np.log(rate_paths[:, 1:] / rate_paths[:, :-1])) / np.sqrt(dt):.2%}")
    
    # 2. Calcul de la NPV du swap
    print("\n2. Calcul de la NPV du swap...")
    swap = InterestRateSwap(swap_params)
    npv_matrix = swap.calculate_npv(rate_paths, market_data.r, time_grid)
    
    print(f"   NPV initiale moyenne: {np.mean(npv_matrix[:, 0]):,.0f}")
    print(f"   NPV finale moyenne: {np.mean(npv_matrix[:, -1]):,.0f}")
    
    # 3. Calcul des métriques d'exposition
    print("\n3. Calcul des métriques d'exposition...")
    cva_calc = CVACalculator(market_data)
    exposure_metrics = cva_calc.calculate_exposure_metrics(npv_matrix)
    
    print(f"   EPE (Expected Positive Exposure): {exposure_metrics['epe']:,.0f}")
    print(f"   PFE max (95% quantile): {exposure_metrics['max_pfe']:,.0f}")
    
    # 4. Simulation des temps de défaut
    print("\n4. Simulation des temps de défaut...")
    lambda_counterparty = market_data.spread_credit / (1 - market_data.recovery_rate)
    lambda_own = lambda_counterparty * 0.5  # Notre risque de défaut plus faible
    
    default_model_cp = DefaultModel(lambda_counterparty, market_data.recovery_rate)
    default_model_own = DefaultModel(lambda_own, market_data.recovery_rate)
    
    counterparty_default_times = default_model_cp.simulate_default_times(T, Nmc)
    own_default_times = default_model_own.simulate_default_times(T, Nmc)
    
    print(f"   Prob. survie contrepartie (5Y): {default_model_cp.survival_probability(T):.2%}")
    print(f"   Prob. survie propre (5Y): {default_model_own.survival_probability(T):.2%}")
    
    # 5. Calcul du CVA
    print("\n5. Calcul du CVA...")
    
    # CVA unilatéral
    cva_unilateral = cva_calc.calculate_unilateral_cva(
        npv_matrix, counterparty_default_times, time_grid
    )
    
    # CVA bilatéral
    cva_bilateral_results = cva_calc.calculate_bilateral_cva(
        npv_matrix, counterparty_default_times, own_default_times, time_grid
    )
    
    print(f"   CVA unilatéral: {cva_unilateral:,.0f}")
    print(f"   CVA bilatéral: {cva_bilateral_results['bilateral_cva']:,.0f}")
    print(f"   DVA: {cva_bilateral_results['dva']:,.0f}")
    
    # 6. Impact du collatéral
    print("\n6. Impact du collatéral...")
    collateral_model = CollateralModel(
        threshold=50000,  # seuil 50k
        minimum_transfer=10000,  # transfert minimum 10k
        haircut=0.02  # décote 2%
    )
    
    npv_collateralized = collateral_model.adjust_exposure_for_collateral(npv_matrix)
    
    cva_with_collateral = cva_calc.calculate_unilateral_cva(
        npv_collateralized, counterparty_default_times, time_grid
    )
    
    print(f"   CVA avec collatéral: {cva_with_collateral:,.0f}")
    print(f"   Réduction CVA: {(cva_unilateral - cva_with_collateral):,.0f}")
    print(f"   Efficacité collatéral: {(1 - cva_with_collateral/cva_unilateral):.1%}")
    
    # 7. Visualisations
    print("\n7. Génération des graphiques...")
    create_plots(time_grid, rate_paths, npv_matrix, exposure_metrics, swap.payment_dates)
    
    print("\nSimulation terminée!")

def create_plots(time_grid: np.ndarray, rate_paths: np.ndarray, 
                npv_matrix: np.ndarray, exposure_metrics: dict, 
                payment_dates: np.ndarray):
    """Crée les graphiques d'analyse"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Graphique 1: Trajectoires de taux
    ax1 = axes[0, 0]
    for i in range(min(100, rate_paths.shape[0])):
        ax1.plot(time_grid, rate_paths[i, :], alpha=0.3, linewidth=0.5)
    ax1.plot(time_grid, np.mean(rate_paths, axis=0), 'r-', linewidth=2, label='Moyenne')
    ax1.set_title('Trajectoires de Taux d\'Intérêt')
    ax1.set_xlabel('Temps (années)')
    ax1.set_ylabel('Taux')
    ax1.legend()
    ax1.grid(True)
    
    # Graphique 2: Distribution NPV finale
    ax2 = axes[0, 1]
    ax2.hist(npv_matrix[:, -1], bins=50, alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(npv_matrix[:, -1]), color='red', linestyle='--', 
               label=f'Moyenne: {np.mean(npv_matrix[:, -1]):,.0f}')
    ax2.set_title('Distribution NPV Finale')
    ax2.set_xlabel('NPV')
    ax2.set_ylabel('Fréquence')
    ax2.legend()
    ax2.grid(True)
    
    # Graphique 3: Profil d'exposition
    ax3 = axes[1, 0]
    ax3.plot(payment_dates, exposure_metrics['ee'], 'b-', linewidth=2, label='EE')
    ax3.plot(payment_dates, exposure_metrics['pfe_95'], 'r-', linewidth=2, label='PFE 95%')
    ax3.set_title('Profil d\'Exposition')
    ax3.set_xlabel('Temps (années)')
    ax3.set_ylabel('Exposition')
    ax3.legend()
    ax3.grid(True)
    
    # Graphique 4: Evolution NPV moyenne
    ax4 = axes[1, 1]
    npv_mean = np.mean(npv_matrix, axis=0)
    npv_std = np.std(npv_matrix, axis=0)
    ax4.plot(payment_dates, npv_mean, 'g-', linewidth=2, label='NPV Moyenne')
    ax4.fill_between(payment_dates, npv_mean - npv_std, npv_mean + npv_std, 
                    alpha=0.3, label='±1 écart-type')
    ax4.set_title('Evolution NPV Moyenne')
    ax4.set_xlabel('Temps (années)')
    ax4.set_ylabel('NPV')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()