"""Class for Hessian (Statistical) Manifolds."""

import geomstats.backend as gs
from geomstats.geometry.embedded_manifold import EmbeddedManifold
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.riemannian_metric import RiemannianMetric


class HessianManifold(EmbeddedManifold):
    def __init__(self, dim, potential):
        super(HessianManifold, self).__init__(
            dim=dim, embedding_manifold=Euclidean(dim))

        self.potential = potential
        self.metric = HessianMetric(dim, potential)


class HessianMetric(RiemannianMetric):
    def __init__(self, dim, potential):
        super(HessianMetric, self).__init__(dim=dim)

        self.potential = potential

    def _potential_grad(self, theta):
        return gs.autograd.elementwise_grad(self.potential)(theta)

    def _potential_hessian(self, theta):
        hessian = gs.autograd.jacobian(self._potential_grad)(theta)
        if theta.ndim == 1:
            return hessian
        return gs.transpose(gs.diagonal(hessian, axis1=0, axis2=2), (2, 0, 1))

    def metric_matrix(self, base_point=None):
        return - self._potential_hessian(base_point)

    def first_type_christoffels(self, base_point):
        if base_point.ndim > 1:
            result = []
            for pt in base_point:
                result.append(
                    -.5 * (gs.autograd.jacobian(self._potential_hessian)(pt)))
            return gs.stack(result)
        return -.5 * (gs.autograd.jacobian(self._potential_hessian)(
            base_point))

    def christoffels(self, base_point):
        first_type = self.first_type_christoffels(base_point)
        inverse_metric = self.inner_product_inverse_matrix(base_point)
        return gs.einsum('...ijl,...kl->...kij', first_type, inverse_metric)
