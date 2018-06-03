import numpy as np

__all__ = [
    'gradient_descent',
    'momentum',
    'nesterov',
    'adagrad',
    'adadelta',
    'rmsprop',
    'rmsprop_nesterov',
    'adam',
    'adamax',
    'nadam',
]


def gradient_descent(f, starting_point, learning_rate):
    """gradient_descent Basic algorithm for adjusting theta parameters
    implemented as generator. Parameters corrections calculated based on
    gradient, which in turn is obtained by taylor polynomial expansion (see
    param f)

    :param f: ExpPoly2D object calculating approimate derivatives based on
    taylor expansion interpolation (or any object returning gradient via taylor
    function) See source code for ExpPoly2D for more information
    :param starting_point: initial theta parameters (for example randomly
    initialized)
    :param learning_rate: Learning rate for the algorithm
    :yields tuple consisting of:
    a) vector of parameters theta
    b) value of function for current parameters
    c) gradient vector consisting of partial derivatives with respect to every
    input parameter

    """
    theta = starting_point.copy()
    while True:
        output, gradient = f.taylor(theta)
        yield theta, output, gradient
        theta -= learning_rate * gradient.reshape(theta.shape)


def momentum(f, starting_point, learning_rate, gamma=0.9):
    """momentum Calculates correction to theta parameters w.r.t. current
    position. Factors in the influence of previous gradient (e.g. faster
    correction of parameter values if both [previous and the current] gradients
    point towards the same direction. If they point in opposite directions the
    step is effectively smaller).

    :param f: ExpPoly2D object calculating approimate derivatives based on
    taylor expansion interpolation
    See source code for more information
    :param starting_point: initial theta parameters (for example randomly
    initialized)
    :param learning_rate: Learning rate for the algorithm
    :param gamma: Multiplier for previous gradient (how big of an effect it has
    on calculation of current error correction)
    :yields tuple consisting of:
    a) vector of parameters theta
    b) value of function for current parameters
    c) gradient vector consisting of partial derivatives with respect to every
    input parameter

    """
    theta = starting_point.copy()
    accumulated_gradient = np.zeros(theta.shape)
    while True:
        output, gradient = f.taylor(theta)
        yield theta, output, gradient
        accumulated_gradient = gamma * accumulated_gradient \
            + learning_rate * gradient.reshape(theta.shape)
        theta -= accumulated_gradient


def nesterov(f, starting_point, learning_rate, gamma=0.9):
    """nesterov Like momentum (see function momentum) but calculates gradient
    w.r.t. to estimation of our future position instead of the current one.

    :param f: ExpPoly2D object calculating approimate derivatives based on
    taylor expansion interpolation
    See source code for more information
    :param starting_point: initial theta parameters (for example randomly
    initialized)
    :param learning_rate: Learning rate for the algorithm
    :param gamma: Multiplier for previous gradient (size of an effect it has
    on calculation of current error correction)
    :yields tuple consisting of:
    a) vector of parameters theta
    b) value of function for current parameters
    c) gradient vector consisting of partial derivatives with respect to every
    input parameter

    """
    # INEFFICIENT, CALCULATING GRADIENT TWICE (?)
    theta = starting_point.copy()
    accumulated_gradient = np.zeros(theta.shape)
    while True:
        output, gradient = f.taylor(theta)
        yield theta, output, gradient
        estimation = theta - gamma * accumulated_gradient
        _, estimation_gradient = f.taylor(estimation)
        accumulated_gradient = gamma * accumulated_gradient \
            + learning_rate * estimation_gradient.reshape(theta.shape)
        theta -= accumulated_gradient


def adagrad(f, starting_point, learning_rate, epsilon=1e-8):
    """adagrad Adapts learning rates to parameters based on their frequency.
    Large updates for infrequent parameters, smaller for the frequent ones.
    Good for sparse data, learning rate becomes infinitesimal as iterations
    count increases (bigger denomiator in each iteration), which may make it
    unusable. Check adadelta for a fix of this problem.

    :param f: ExpPoly2D object calculating approimate derivatives based on
    taylor expansion interpolation
    See source code for more information
    :param starting_point: initial theta parameters (for example randomly
    initialized)
    :param learning_rate: Learning rate for the algorithm
    (defaulted to 0.01 as is seen in most research papers)
    :param epsilon: number to counter possible numerical instability of the
    method (defaulted to 1e-8, reason as in [learning_rate])
    :yields tuple consisting of:
    a) vector of parameters theta
    b) value of function for current parameters
    c) gradient vector consisting of partial derivatives with respect to every
    input parameter

    """
    theta = starting_point.copy()
    accumulated_gradients = np.zeros(theta.shape)
    while True:
        output, gradient = f.taylor(theta)
        yield theta, output, gradient
        reshaped_gradient = gradient.reshape(theta.shape)
        accumulated_gradients += np.sqrt(np.abs(reshaped_gradient))
        theta -= ((learning_rate / np.sqrt(accumulated_gradients + epsilon)) *
                  reshaped_gradient)


def adadelta(f, starting_point, gamma=0.9, epsilon=1e-8):
    """adadelta.

    NOT WORKING RIGHT NOW

    Like adagrad but fixes problem of infinitesimal learning rate by calculating
    moving average of previous gradients instead of accumulating them.

    Similiar to RMSProp, but does not need the learning rate at all.
    Forementioned calculated by moving average of parameters taken to the power
    of 2.

    :param f: ExpPoly2D object calculating approimate derivatives based on
    taylor expansion interpolation
    See source code for more information
    :param starting_point: initial theta parameters (for example randomly
    initialized)
    :param gamma: Multiplier for previous gradient (size of an effect it has
    on calculation of current error correction)
    :param epsilon: number to counter possible numerical instability of the
    method (defaulted to 1e-8, reason as in [learning_rate])
    :yields tuple consisting of:
    a) vector of parameters theta
    b) value of function for current parameters
    c) gradient vector consisting of partial derivatives with respect to every
    input parameter

    """

    theta = starting_point.copy()
    average_gradients = np.zeros(theta.shape)
    average_parameters = np.zeros(theta.shape)
    while True:
        output, gradient = f.taylor(theta)
        yield theta, output, gradient

        average_parameters = gamma * \
            average_parameters + (1. - gamma) * (theta**2)

        gradient_reshaped = gradient.reshape(theta.shape)
        average_gradients = gamma * average_gradients + \
            (1. - gamma) * gradient_reshaped**2

        theta -= (np.sqrt(average_parameters + epsilon)
                  / np.sqrt(average_gradients + epsilon)) \
            * gradient_reshaped


def rmsprop(f, starting_point, learning_rate=1e-3, gamma=0.9, epsilon=1e-8):
    """rmsprop Like adadelta, the only difference is the existence of learning
    rate (this method DOES NOT calculate moving average of parameters)

    Created independently around the same time span, proposed by Hilton in one
    of his coursers

    :param f: ExpPoly2D object calculating approimate derivatives based on
    taylor expansion interpolation. See source code for more information
    :param starting_point: initial theta parameters (for example randomly
    initialized)
    :param learning_rate: Learning rate for the algorithm
    :param gamma: Multiplier for previous gradient (size of an effect it has
    on calculation of current error correction)
    :param epsilon: number to counter possible numerical instability of the
    method (defaulted to 1e-8, reason as in [learning_rate])
    :yields tuple consisting of:
    a) vector of parameters theta
    b) value of function for current parameters
    c) gradient vector consisting of partial derivatives with respect to every
    input parameter

    """
    theta = starting_point.copy()
    average_gradients = np.zeros(theta.shape)
    while True:
        output, gradient = f.taylor(theta)
        yield theta, output, gradient
        gradient_reshaped = gradient.reshape(theta.shape)
        average_gradients = (1 - gamma) * average_gradients + \
            gamma * gradient_reshaped**2
        theta -= (learning_rate / (np.sqrt(average_gradients + epsilon))) \
            * gradient_reshaped


def rmsprop_nesterov(f,
                     starting_point,
                     learning_rate=1e-3,
                     gamma=0.9,
                     epsilon=1e-8):
    """rmsprop_nesterov BUGGED BETA Version of rmsprop using nesterov update
    rule.

    Created independently around the same time span, proposed by Hilton in one
    of his coursers

    :param f: ExpPoly2D object calculating approimate derivatives based on
    taylor expansion interpolation
    See source code for more information
    :param starting_point: initial theta parameters (for example randomly
    initialized)
    :param learning_rate: Learning rate for the algorithm
    :param gamma: Multiplier for previous gradient (size of an effect it has
    on calculation of current error correction)
    :param epsilon: number to counter possible numerical instability of the
    method (defaulted to 1e-8, reason as in [learning_rate])
    :yields tuple consisting of:
    a) vector of parameters theta
    b) value of function for current parameters
    c) gradient vector consisting of partial derivatives with respect to every
    input parameter

    """
    theta = starting_point.copy()
    average_gradients = np.zeros(theta.shape)
    while True:
        output, gradient = f.taylor(theta)
        yield theta, output, gradient
        estimation = theta - (gamma * average_gradients)
        _, estimation_gradient = f.taylor(estimation)
        estimation_gradient_reshaped = estimation_gradient.reshape(theta.shape)

        average_gradients = (1 - gamma) * average_gradients + \
            gamma * estimation_gradient_reshaped**2
        theta -= (learning_rate / (np.sqrt(average_gradients + epsilon))
                  ) * estimation_gradient_reshaped


def adam(f,
         starting_point,
         learning_rate=0.001,
         beta1=0.9,
         beta2=0.999,
         epsilon=1e-8):
    """adam Adaptive Moments algorithm created by Google.

    Includes learning rates for each of the parameters, just like adagrad and
    their derivations (adagrad, RMSProp) and, additionaly, keeps exponentially
    decaying average of past gradients similiarly to momentum.

    :param f: ExpPoly2D object calculating approimate derivatives based on
    taylor expansion interpolation
    See source code for more information
    :param starting_point: initial theta parameters (for example randomly
    initialized)
    :param learning_rate: Learning rate for the algorithm
    :param beta1: Multiplier for momentum (exponentially decaying average past
    gradients)
    :param beta2: Multiplier for parameters rate (see adagrad/adadelta/rmsprop)
    :param epsilon: number to counter possible numerical instability of the
    method (defaulted to 1e-8, reason as in [learning_rate])
    :yields tuple consisting of:
    a) vector of parameters theta
    b) value of function for current parameters
    c) gradient vector consisting of partial derivatives with respect to every
    input parameter

    """
    theta = starting_point.copy()
    # LIKE RMSProp or Adadelta
    average_gradients_squared = np.zeros(theta.shape)
    # LIKE MOMENTUM
    average_gradients = np.zeros(theta.shape)

    while True:
        output, gradient = f.taylor(theta)
        yield theta, output, gradient
        gradient_reshaped = gradient.reshape(theta.shape)

        average_gradients = beta1 * average_gradients + \
            (1. - beta1) * gradient_reshaped
        average_gradients_squared = beta2 * average_gradients_squared \
            + (1. - beta2) * gradient_reshaped**2

        theta -= learning_rate \
            / (np.sqrt(average_gradients_squared / (1 - beta2)) + epsilon) \
            * ((average_gradients) / (1 - beta1))


def adamax(f, starting_point, learning_rate=1e-2, beta1=0.9, beta2=0.999):
    """adamax Adaptive Moments (adam) modification (same research paper)

    Modification: learning rates are either based on current gradient or the
    previous ones (like adam, adagrad etc.). Decisive is the maximum norm of
    them, the one with higher is chosen.

    NOTE: no epsilon is needed, as this function is not biased towards zero as
    much as adam (see original research paper).

    :param f: ExpPoly2D object calculating approimate derivatives based on
    taylor expansion interpolation
    See source code for more information
    :param starting_point: initial theta parameters (for example randomly
    initialized)
    :param learning_rate: Learning rate for the algorithm
    :param beta1: Multiplier for momentum (exponentially decaying average past
    gradients)
    :param beta2: Multiplier for parameters rate (see adagrad/adadelta/rmsprop)
    :yields tuple consisting of:
    a) vector of parameters theta
    b) value of function for current parameters
    c) gradient vector consisting of partial derivatives with respect to every
    input parameter

    """
    theta = starting_point.copy()
    # LIKE RMSProp or Adadelta
    previous_gradient = np.zeros(theta.shape)
    # LIKE MOMENTUM
    average_gradients = np.zeros(theta.shape)

    while True:
        output, gradient = f.taylor(theta)
        yield theta, output, gradient

        gradient_reshaped = gradient.reshape(theta.shape)
        previous_gradient = np.maximum(beta2 * previous_gradient,
                                       np.abs(gradient_reshaped))
        average_gradients = beta1 * average_gradients \
            + (1. - beta1) * gradient_reshaped

        theta -= (learning_rate / previous_gradient) \
            * ((average_gradients) / (1 - beta1))


def nadam(f,
          starting_point,
          learning_rate=0.001,
          beta1=0.9,
          beta2=0.999,
          epsilon=1e-8):
    """nadam Modification of Adaptive Moments algorithm.

    For more detailed description see: [adam]

    Modification to adam: calculates gradient w.r.t. estimation of current
    position instead of the previous one

    :param f: ExpPoly2D object calculating approimate derivatives based on
    taylor expansion interpolation
    See source code for more information
    :param starting_point: initial theta parameters (for example randomly
    initialized)
    :param learning_rate: Learning rate for the algorithm
    :param beta1: Multiplier for momentum (exponentially decaying average past
    gradients)
    :param beta2: Multiplier for parameters rate (see adagrad/adadelta/rmsprop)
    :param epsilon: number to counter possible numerical instability of the
    method (defaulted to 1e-8, reason as in [learning_rate])
    :yields tuple consisting of:
    a) vector of parameters theta
    b) value of function for current parameters
    c) gradient vector consisting of partial derivatives with respect to every
    input parameter

    """

    theta = starting_point.copy()
    # LIKE RMSProp or Adadelta
    average_gradients_squared = np.zeros(theta.shape)
    # LIKE MOMENTUM
    average_gradients = np.zeros(theta.shape)

    while True:
        output, gradient = f.taylor(theta)
        yield theta, output, gradient

        reshaped_gradient = gradient.reshape(theta.shape)
        average_gradients_squared = beta2 * average_gradients_squared \
            + (1. - beta2) * reshaped_gradient**2
        average_gradients = beta1 * reshaped_gradient \
            + (1. - beta2) * reshaped_gradient

        theta -= learning_rate \
            / (np.sqrt(average_gradients_squared / (1 - beta2)) + epsilon) \
            * ((average_gradients) / (1 - beta1))
