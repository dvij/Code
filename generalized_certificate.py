from abc import ABC, abstractmethod
import logging
from typing import Any, List


class GeneralizedCertificate(ABC):
    def __init__(self, data_distribution: Any, ):
        self._data_distribution = data_distribution
        return
    
    @abstractmethod
    def lambda_fun(self, theta: Any, **kwargs) -> float:
        """Implement the mapping \lambda(\theta)"""
        raise NotImplementedError

    @abstractmethod
    def certificate(self, **kwargs) -> float:
        """Implement the mapping \sup_{\theta, z_adv} E_{\theta' ~ \epsilon\delta(F(\theta, z)) + F(\theta, \zadv)}[\lambda(\theta')] - \lambda(\theta) + l_adv(\theta)"""
        raise NotImplementedError
    
    def optimize_certificate(self,) -> tuple[float, Any]:
        """Implement optimize certificate with respect to lambda kwargs."""
        raise NotImplementedError
    
