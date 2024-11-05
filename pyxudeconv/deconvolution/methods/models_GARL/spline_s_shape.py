import torch


# parametrization of a symmetric 4-knot spline with s-shape
#      /\▁▁▁▁
#▔▔▔▔\/

def s_shape(x, mu, beta, gamma):
    x_sg = torch.sign(x)
    x_abs = torch.abs(x)

    return(x_sg * (torch.relu(torch.minimum(mu * x_abs, beta - gamma * x_abs)))) #+  gamma * torch.relu(x_abs - beta / gamma))

def s_shape_integrate(x, mu, beta, gamma):
    theta = beta/(gamma + mu)
    b = beta/gamma
    c3 = 0.
    c2 = 0.5*(mu + gamma)*theta**2 - beta*theta
    c1 = c2 + 0.5*beta**2/gamma# - 0.5*beta**2/gamma

    xabs = torch.abs(x)
    return(c1*(xabs > b) + (c2 + beta*xabs - 0.5*gamma*x**2)*(xabs > theta)*(xabs <= b) + (c3 + mu*x**2/2.)*(xabs <= theta))


def s_shape_grad(x, mu, beta, gamma):
    x_abs = torch.abs(x)
    o = torch.ones_like(x)

    grad_x = mu * torch.heaviside(x_abs, o) - mu * torch.heaviside(x_abs - beta / (gamma + mu), o)
    grad_x += - gamma*torch.heaviside(x_abs - beta / (gamma + mu), o) + gamma*torch.heaviside(x_abs - beta / gamma, o)

    return(grad_x)

