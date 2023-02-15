from .CTA import CTA, CTA_Autoencoder


def build_network(net_name, ae_net=None):
    """Builds the neural network."""

    implemented_networks = ('HYDICE_urban_CTA')
    assert net_name in implemented_networks

    net = None

    if net_name == 'HYDICE_urban_CTA':
        net = CTA((1,1), 175)

    return net


def build_autoencoder(net_name):
    """Builds the corresponding autoencoder network."""

    implemented_networks = ('HYDICE_urban_CTA')

    assert net_name in implemented_networks

    ae_net = None

    if net_name == 'HYDICE_urban_CTA':
        ae_net = CTA_Autoencoder((1,1), 175)
    return ae_net
