import numpy as np
import tensorflow as tf
import warnings
import torch.nn.functional as F
import random
import torch.autograd as autograd
import torch
import os
EPSI=1e-10
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def get_squared_dist(
        x, y=None, scale=None, z_score=False, mode='xxxyyy', name='squared_dist',
        do_summary=False, scope_prefix=''):
    """ This function calculates the pairwise distance between x and x, x and y, y and y
    Warning: when x, y has mean far away from zero, the distance calculation is not accurate; use get_dist_ref instead
    :param x: batch_size-by-d matrix
    :param y: batch_size-by-d matrix
    :param scale: 1-by-d vector, the precision vector. dxy = x*scale*y
    :param z_score:
    :param mode: 'xxxyyy', 'xx', 'xy', 'xxxy'
    :param name:
    :param do_summary:
    :param scope_prefix: summary scope prefix
    :return:
    """
    if len(list(x.size()))>2:
        raise AttributeError('get_dist: Input must be a matrix.')
    if y is None:
        mode = 'xx'
    if z_score:
        if y is None:
            mu = torch.mean(x, dim=0, keepdims=True)
            x = x - mu
        else:
            mu = torch.mean(torch.cat((x, y), dim=0), dim=0, keepdims=True)
            x = x - mu
            y = y - mu
    if mode in ['xx', 'xxxy', 'xxxyyy']:
        if scale is None:
            xxt = torch.matmul(x, x.t())
            # xxt = tf.matmul(x, x, transpose_b=True)  # [xi_xi, xi_xj; xj_xi, xj_xj], batch_size-by-batch_size
        else:
            xxt = torch.matmul(x * scale, x.t())

        dx = torch.diag(xxt)  # [xxt], [batch_size]
        if torch.cuda.is_available():
            # device = dx.get_device()
            dist_xx = torch.max(torch.unsqueeze(dx, dim=1) - 2.0 * xxt + torch.unsqueeze(dx, dim=1),
                                    torch.Tensor([0.0]).cuda())
        else:
            dist_xx = torch.max(torch.unsqueeze(dx, dim=1) - 2.0 * xxt + torch.unsqueeze(dx, dim=1),
                                torch.Tensor([0.0]))

        if mode == 'xx':
            return dist_xx

        elif mode == 'xxxy':  # estimate dy without yyt
            if scale is None:
                xyt = torch.matmul(x, y.t())
                dy = torch.sum(torch.mul(y, y), dim=1)
            else:
                xyt = torch.matmul(x * scale, y.t())
                dy = torch.sum(torch.mul(y * scale, y), dim=1)
            dist_xy = torch.max(torch.unsqueeze(dx, dim=1) - 2.0 * xyt + torch.unsqueeze(dy, dim=0), 0.0)
            return dist_xx, dist_xy
        elif mode == 'xxxyyy':
            if scale is None:
                xyt = torch.matmul(x, y.t())
                yyt = torch.matmul(y, y.t())
            else:
                xyt = torch.matmul(x * scale, y.t())
                yyt = torch.matmul(y * scale, y.t())
            dy = torch.diag(yyt)
            if torch.cuda.is_available():
                # device = dx.get_device()
                dist_xy = torch.max(torch.unsqueeze(dx, dim=1) - 2.0 * xyt + torch.unsqueeze(dy, dim=0),
                                    torch.Tensor([0.0]).cuda())
                # device = dy.get_device()
                dist_yy = torch.max(torch.unsqueeze(dy, dim=1) - 2.0 * yyt + torch.unsqueeze(dy, dim=0),
                                    torch.Tensor([0.0]).cuda())
            else:
                # device = dx.get_device()
                dist_xy = torch.max(torch.unsqueeze(dx, dim=1) - 2.0 * xyt + torch.unsqueeze(dy, dim=0),
                                    torch.Tensor([0.0]))
                dist_yy = torch.max(torch.unsqueeze(dy, dim=1) - 2.0 * yyt + torch.unsqueeze(dy, dim=0),
                                    torch.Tensor([0.0]))
            return dist_xx, dist_xy, dist_yy
    elif mode == 'xy':
        if scale is None:
            dx = torch.sum(torch.mul(x, x), dim=1)
            dy = torch.sum(torch.mul(y, y), dim=1)
            xyt = torch.matmul(x, y.t())
            # print(torch.mul(x, x))
            # print(dx)
            # print(dy)
            # print(xyt)
        else:
            dx = torch.sum(torch.mul(x * scale, x), dim=1)
            dy = torch.sum(torch.mul(y * scale, y), dim=1)
            xyt = torch.matmul(x * scale, y.t())
        if torch.cuda.is_available():
            # device = dx.get_device()
            dist_xy = torch.max(torch.unsqueeze(dx, dim=1) - 2.0 * xyt + torch.unsqueeze(dy, dim=0), torch.Tensor([0]).cuda())
        else:
            dist_xy = torch.max(torch.unsqueeze(dx, dim=1) - 2.0 * xyt + torch.unsqueeze(dy, dim=0), torch.Tensor([0.0]))

        # print(torch.unsqueeze(dx, dim=1))
        # print(torch.unsqueeze(dy, dim=0))
        # print(2.0 * xyt)
        # print(torch.unsqueeze(dx, dim=1) - 2.0 * xyt + torch.unsqueeze(dy, dim=0))
        return dist_xy
    else:
        raise AttributeError('Mode {} not supported'.format(mode))
def witness_g(dist_zx, dist_zy, sigma=2.0, name='witness', do_summary=False, scope_prefix=''):
    """ This function calculates the witness function f(z) = Ek(x, z) - Ek(y, z) based on Gaussian kernel
    :param dist_zx:
    :param dist_zy:
    :param sigma:
    :param name:
    :param do_summary:
    :param scope_prefix:
    :return:
    """
    # k_zx = tf.exp(-dist_zx / (2.0 * sigma), name='k_zx')
    # k_zy = tf.exp(-dist_zy / (2.0 * sigma), name='k_zy')
    #
    # e_kx = tf.reduce_mean(k_zx, axis=1)
    # e_ky = tf.reduce_mean(k_zy, axis=1)
    k_zx = torch.exp(-dist_zx / (2.0*sigma**2))
    k_zy = torch.exp(-dist_zy / (2.0*sigma**2))

    # print(k_zx)
    # print(k_zy)

    e_kx = torch.mean(k_zx,dim=1)
    e_ky = torch.mean(k_zy,dim=1)
    # print(e_kx)
    # print(e_ky)
    witness = e_kx - e_ky

    return witness

def witness_mix_g(dist_zx, dist_zy, sigma=None, name='witness', do_summary=False):
    """ This function calculates the witness function f(z) = Ek(x, z) - Ek(y, z) based on
    a list of t-distribution kernels.
    :param dist_zx:
    :param dist_zy:
    :param sigma:
    :param name:
    :param do_summary:
    :return:
    """
    num_sigma = len(sigma)
    witness = 0.0
    for i in range(num_sigma):
        wit_i = witness_g(
            dist_zx, dist_zy, sigma=sigma[i], name='d{}'.format(i), do_summary=do_summary)
        witness = witness + wit_i
        # print(wit_i)

    return witness

def jacobian_squared_frobenius_norm(y, x, name='J_fnorm', do_summary=False):
    """ This function calculates the squared frobenious norm, e.g. sum of square of all elements in Jacobian matrix
        :param y: batch_size-by-d matrix
        :param x: batch_size-by-s tensor
        :param name:
        :param do_summary:
        :return:
        """
    dim_list = list(y.size())
    batch_size, d = dim_list[0],dim_list[1]
    # sfn - squared frobenious norm
    if d == 1:
        y.requires_grad = True
        grads = autograd.grad(y,x,retain_graph=True,create_graph=True)
        gradss = grads[0]
        jaco_sfn = torch.sum(torch.square(torch.reshape(gradss,[batch_size,-1])),dim=1)
        # jaco_sfn = tf.reduce_sum(tf.square(tf.reshape(tf.gradients(y, x)[0], [batch_size, -1])), axis=1)
    else:
        y.requires_grad = True
        jaco_sfn = torch.sum(
            torch.stack(
                [torch.sum(
                    torch.square(torch.reshape(autograd.grad(y[:, i], x,create_graph=True,retain_graph=True)[0], [batch_size, -1])),  # b-vector
                    dim=1) for i in range(d)],
                dim=0),  # d-by-b
            dim=0)  # b-vector

    # if do_summary:
    #     with tf.name_scope(None):  # return to root scope to avoid scope overlap
    #         tf.summary.histogram('Jaco_sfn', jaco_sfn)

    return jaco_sfn

# if __name__ == '__main__':
#     a = torch.Tensor([[1,2,3],[4,5,6]])
#     c = torch.Tensor([[2,3,4],[5,6,7]])
#
#     print(list(a.size()))
#     ac = (get_squared_dist(a,c,mode='xy'))
#     ca = (get_squared_dist(a,a,mode='xy'))
#     print(witness_g(ac,ca))
def jacobian(y, x, name='jacobian'):
    """ This function calculates the jacobian matrix: dy/dx and returns a list
    :param y: batch_size-by-d matrix
    :param x: batch_size-by-s tensor
    :param name:
    :return:
    """
    # with tf.name_scope(name):
    dim_list = list(y.size())
    batch_size, d = dim_list[0], dim_list[1]
    if d == 1:
        y.requires_grad = True
        grads = autograd.grad(y, x, retain_graph=True, create_graph=True)
        gradss = grads[0]
        # jaco_sfn = torch.sum(torch.square(torch.reshape(gradss, [batch_size, -1])), dim=1)
        return torch.reshape(gradss, [batch_size, -1])  # b-by-s
    else:
        dd = torch.stack(
                    [torch.reshape(autograd.grad(y[:, i], x,retain_graph=True,create_graph=True)[0], [batch_size, -1]) for i in range(d)], dim=0)  # d-b-s
        return dd.permute(1, 0, 2)  # b-d-s tensor

def mmd_logisitic_loss(s_recon,s_x):
    loss_dis = torch.mean(F.softplus(s_recon) + F.softplus(-s_x))
    loss_gen = torch.mean(F.softplus(-s_recon))
    return loss_dis,loss_gen

def mmd_hinge_loss(s_recon,s_x):
    loss_dis = torch.mean(
        F.relu(1.0 + s_recon)) + torch.mean(F.relu(1.0 - s_x))
    loss_gen = torch.mean(-s_recon)
    return loss_dis,loss_gen

def mmd_wasserstein_loss(s_recon,s_x,dis_penalty):
    loss_gen = torch.mean(s_x) - torch.mean(s_recon)
    loss_dis = - loss_gen + dis_penalty

    return loss_dis,loss_gen

def matrix_mean_wo_diagonal(matrix, num_row, num_col=None, name='mu_wo_diag'):
    """ This function calculates the mean of the matrix elements not in the diagonal
    2018.4.9 - replace tf.diag_part with tf.matrix_diag_part
    tf.matrix_diag_part can be used for rectangle matrix while tf.diag_part can only be used for square matrix
    :param matrix:
    :param num_row:
    :type num_row: float
    :param num_col:
    :type num_col: float
    :param name:
    :return:
    """
    if num_col is None:
        mu = (torch.sum(matrix) - torch.sum(torch.diag(matrix))) / (num_row * (num_row - 1.0))
    else:
        mu = (torch.sum(matrix) - torch.sum(torch.diag(matrix))) \
                 / (num_row * num_col - max(num_col,num_row))

    return mu


def row_mean_wo_diagonal(matrix, num_col, name='mu_wo_diag'):
    """ This function calculates the mean of each row of the matrix elements excluding the diagonal

    :param matrix:
    :param num_col:
    :type num_col: float
    :param name:
    :return:
    """
    # with tf.name_scope(name):
    return (torch.sum(matrix, dim=1) - torch.diag(matrix)) / (num_col - 1.0)

def mmd_g(
        dist_xx, dist_xy, dist_yy, batch_size, sigma=1.0, var_target=None, upper_bound=None, lower_bound=None,
        name='mmd', do_summary=False, scope_prefix='', custom_weights=None):
    """This function calculates the maximum mean discrepancy with Gaussian distribution kernel
        The kernel is taken from following paper:
        Li, C.-L., Chang, W.-C., Cheng, Y., Yang, Y., & Póczos, B. (2017).
        MMD GAN: Towards Deeper Understanding of Moment Matching Network.
        :param dist_xx:
        :param dist_xy:
        :param dist_yy:
        :param batch_size:
        :param sigma:
        :param var_target: if sigma is trainable, var_target contain the target for sigma
        :param upper_bound: bounds for pairwise distance in mmd-g.
        :param lower_bound:
        :param name:
        :param do_summary:
        :param scope_prefix:
        :param custom_weights: weights for loss in mmd, default is [2.0, 1.0], custom[0] - custom[1] = 1.0
        :type custom_weights: list
        :return:
        """
    if lower_bound is None:
        k_xx = torch.exp(-dist_xx / (2.0 * sigma ** 2))
        k_yy = torch.exp(-dist_yy / (2.0 * sigma ** 2))
    else:
        if torch.cuda.is_available():
            # device = dist_xx.get_device()
            k_xx = torch.exp(-torch.max(dist_xx, torch.Tensor([lower_bound]).cuda()) / (2.0 * sigma ** 2))
            k_yy = torch.exp(-torch.max(dist_yy, torch.Tensor([lower_bound]).cuda()) / (2.0 * sigma ** 2))
        else:
            k_xx = torch.exp(-torch.max(dist_xx, torch.Tensor([lower_bound])) / (2.0 * sigma ** 2))
            k_yy = torch.exp(-torch.max(dist_yy, torch.Tensor([lower_bound])) / (2.0 * sigma ** 2))


    if upper_bound is None:
        k_xy = torch.exp(-dist_xy / (2.0 * sigma ** 2))
    else:
        if torch.cuda.is_available():
            # device = dist_xy.get_device()
            k_xy = torch.exp(-torch.min(dist_xy, torch.Tensor([upper_bound]).cuda()) / (2.0 * sigma ** 2))
        else:
            k_xy = torch.exp(-torch.min(dist_xy, torch.Tensor([upper_bound])) / (2.0 * sigma ** 2))
    m = batch_size
    e_kxx = matrix_mean_wo_diagonal(k_xx,m)
    e_kxy = matrix_mean_wo_diagonal(k_xy,m)
    e_kyy = matrix_mean_wo_diagonal(k_yy,m)

    if var_target is None:
        if custom_weights is None:
            mmd = e_kxx + e_kyy - 2.0 * e_kxy
            return mmd
        else:
            # note that here kyy is for the real data!
            assert custom_weights[0] - custom_weights[1] == 1.0, 'w[0]-w[1] must be 1'
            mmd1 = e_kxx + e_kyy - 2.0 * e_kxy
            mmd2 = custom_weights[0] * e_kxy - e_kxx - custom_weights[1] * e_kyy
            return mmd1, mmd2
    else:
        mmd = e_kxx + e_kyy - 2.0 * e_kxy
        var = e_kxx + e_kyy + 2.0 * e_kxy
        loss_sigma = torch.square(var - var_target)

        return mmd,loss_sigma

def mmd_g_bounded(
        dist_xx, dist_xy, dist_yy, batch_size, sigma=1.0, var_target=None, upper_bound=None, lower_bound=None,
        name='mmd', do_summary=False, scope_prefix='', custom_weights=None):
    """This function calculates the maximum mean discrepancy with Gaussian distribution kernel
    The kernel is taken from following paper:
    Li, C.-L., Chang, W.-C., Cheng, Y., Yang, Y., & Póczos, B. (2017).
    MMD GAN: Towards Deeper Understanding of Moment Matching Network.
    :param dist_xx:
    :param dist_xy:
    :param dist_yy:
    :param batch_size:
    :param sigma:
    :param var_target: if sigma is trainable, var_target contain the target for sigma
    :param upper_bound:
    :param lower_bound:
    :param name:
    :param do_summary:
    :param scope_prefix:
    :param custom_weights: weights for loss in mmd, default is [2.0, 1.0], custom[0] - custom[1] = 1.0
    :type custom_weights: list
    :return:
    """
    k_xx = torch.exp(-dist_xx / (2.0 * sigma ** 2))
    k_yy = torch.exp(-dist_yy / (2.0 * sigma ** 2))
    k_xy = torch.exp(-dist_xy / (2.0 * sigma ** 2))

    # in rep loss, custom_weights[0] - custom_weights[1] = 1
    if torch.cuda.is_available():
        # device = dist_xx.get_device()
        k_xx_b = torch.exp(-torch.max(dist_xx, torch.Tensor([lower_bound]).cuda())/(2.0*sigma**2))
    else:
        k_xx_b = torch.exp(-torch.max(dist_xx, torch.Tensor([lower_bound]))/(2.0*sigma**2))

    if custom_weights[0] > 0:
        if torch.cuda.is_available():
            # device = dist_xy.get_device()
            k_xy_b = torch.exp(-torch.min(dist_xy,torch.Tensor([upper_bound]).cuda()) / (2.0 * sigma ** 2))
        else:
            k_xy_b = torch.exp(-torch.min(dist_xy,torch.Tensor([upper_bound])) / (2.0 * sigma ** 2))
    else:
        k_xy_b = k_xy

    if custom_weights[1] > 0:  # the original mmd-g
        if torch.cuda.is_available():
            # device = dist_yy.get_device()
            k_yy_b = torch.exp(-torch.max(dist_yy, torch.Tensor([lower_bound]).cuda()) / (2.0 * sigma ** 2))
        else:
            k_yy_b = torch.exp(-torch.max(dist_yy, torch.Tensor([lower_bound])) / (2.0 * sigma ** 2))

    else:  # the repulsive mmd-g
        if torch.cuda.is_available():
            # device = dist_yy.get_device()
            k_yy_b = torch.exp(-torch.min(dist_yy, torch.Tensor([upper_bound]).cuda()) / (2.0 * sigma ** 2))
        else:
            k_yy_b = torch.exp(-torch.min(dist_yy, torch.Tensor([upper_bound])) / (2.0 * sigma ** 2))

    m = batch_size
    e_kxx = matrix_mean_wo_diagonal(k_xx, m)
    e_kxy = matrix_mean_wo_diagonal(k_xy, m)
    e_kyy = matrix_mean_wo_diagonal(k_yy, m)
    e_kxx_b = matrix_mean_wo_diagonal(k_xx_b, m)
    e_kyy_b = matrix_mean_wo_diagonal(k_yy_b, m)
    e_kxy_b = matrix_mean_wo_diagonal(k_xy_b, m) if custom_weights[0] < 0 else e_kxy

    if var_target is None:
        if custom_weights is None:
            mmd = e_kxx + e_kyy - 2.0 * e_kxy
            return mmd
        else:
            assert custom_weights[0] - custom_weights[1] == 1.0, 'w[0]-w[1] must be 1'
            mmd1 = e_kxx + e_kyy - 2.0 * e_kxy
            mmd2 = custom_weights[0] * e_kxy_b - e_kxx_b - custom_weights[1] * e_kyy_b
            return mmd1, mmd2
    else:
        mmd = e_kxx + e_kyy - 2.0 * e_kxy
        var = e_kxx + e_kyy + 2.0 * e_kxy
        loss_sigma = torch.square(var - var_target)

        return mmd, loss_sigma


def mixture_mmd_g(
        dist_xx, dist_xy, dist_yy, batch_size, sigma=None, var_targets=None, name='mmd_g',
        do_summary=False, scope_prefix=''):
    """ This function calculates the maximum mean discrepancy with a list of Gaussian distribution kernel
    :param dist_xx:
    :param dist_xy:
    :param dist_yy:
    :param batch_size:
    :param sigma:
    :type sigma: list
    :param var_targets: if sigma is trainable, var_targets contain the target for each sigma
    :type var_targets: list
    :param name:
    :param do_summary:
    :param scope_prefix:
    :return:
    """
    num_sigma = len(sigma) if isinstance(sigma, list) else len(var_targets)
    mmd = 0.0

    if var_targets is None:
        for i in range(num_sigma):
            mmd_i = mmd_g(
                dist_xx, dist_xy, dist_yy, batch_size, sigma=sigma[i],
                name='d{}'.format(i), do_summary=do_summary, scope_prefix=scope_prefix + name + '/')
            mmd = mmd + mmd_i

        return mmd
    else:
        loss_sigma = 0.0
        for i in range(num_sigma):
            mmd_i, loss_i = mmd_g(
                dist_xx, dist_xy, dist_yy, batch_size, sigma=sigma[i], var_target=var_targets[i],
                name='d{}'.format(i), do_summary=do_summary, scope_prefix=scope_prefix + name + '/')
            mmd = mmd + mmd_i
            loss_sigma = loss_sigma + loss_i

        return mmd,loss_sigma

def mmd_g_loss(s_recon,s_x,dis_penalty=None,sigma=[1,np.sqrt(2),2,np.sqrt(8),4]):
    """ maximum mean discrepancy with gaussian kernel
            """
    # calculate pairwise distance
    batch_size = s_recon.size()[0]
    dist_gg, dist_gd, dist_dd = get_squared_dist(s_recon,s_x,z_score=False)
    # mmd
    loss_gen = mixture_mmd_g(
        dist_gg, dist_gd, dist_dd, batch_size, sigma=sigma,
        name='mmd_g')
    loss_dis = -loss_gen
    if dis_penalty is not None:
        loss_dis = loss_dis + dis_penalty

    return loss_dis,loss_gen

def mmd_g_bound_loss(s_recon,s_x,dis_penalty=None):
    """ maximum mean discrepancy with gaussian kernel and bounds on dxy
            :return:
            """
    # calculate pairwise distance
    batch_size = s_recon.size()[0]
    dist_gg, dist_gd, dist_dd = get_squared_dist(
        s_recon, s_x, z_score=False)

    # mmd
    loss_gen = mmd_g(
        dist_gg, dist_gd, dist_dd, batch_size, sigma=1.0,
        name='mmd_g', scope_prefix='')
    mmd_b = mmd_g(
        dist_gg, dist_gd, dist_dd, batch_size, sigma=1.0, upper_bound=4, lower_bound=0.25,
        name='mmd_g_b',scope_prefix='')
    loss_dis = -mmd_b
    if dis_penalty is not None:
        loss_dis = loss_dis + dis_penalty

    return loss_dis,loss_gen
def mat_slice(mat, row_index, col_index=None, name='slice'):
    """ This function gets mat[index, index] where index is either bool or int32.
      Note that:
          if index is bool, output size is typically smaller than mat unless each element in index is True
          if index is int32, output can be any size.
      :param mat:
      :param row_index:
      :param col_index:
      :param name;
      :return:
      """
    if col_index is None:
        col_index = row_index
    if row_index.dtype != col_index.dtype:
        raise AttributeError('dtype of row-index and col-index do not match.')
    if row_index.dtype == np.int:
        # torch.gather()
        row_index = row_index.squeeze()
        col_index = col_index.squeeze()
        # torch.index_select(mat,dim=0,index=torch.LongTensor(row_index))
        # torch.gather(mat,dim=0,index=torch.LongTensor(row_index))
        return torch.index_select(torch.index_select(mat,dim=0,index=torch.LongTensor(row_index)),dim=1,index=torch.LongTensor(col_index))
    elif row_index.dtype == np.bool:
        row_id = np.argwhere(row_index==True)
        row_id = row_id.squeeze()
        col_id = np.argwhere(col_index==True)
        col_id = col_id.squeeze()

        return torch.index_select(torch.index_select(mat,dim=0,index=torch.LongTensor(row_id)),dim=1,index=torch.LongTensor(col_id))
    else:
        raise AttributeError('Type of index is: {}; expected either tf.int32 or tf.bool'.format(row_index.dtype))


def slice_pairwise_distance(pair_dist, batch_size=None, indices=None):
    """ This function slice pair-dist into smaller pairwise distance matrices
    :param pair_dist: 2batch_size-by-2batch_size pairwise distance matrix
    :param batch_size:
    :param indices:
    :return:
    """
    if indices is None:
        dist_g1 = pair_dist[0:batch_size, 0:batch_size]
        dist_g2 = pair_dist[batch_size:, batch_size:]
        dist_g1g2 = pair_dist[0:batch_size, batch_size:]
    else:
        mix_group_1 = torch.cat((indices, torch.logical_not(indices)), dim=0)
        mix_group_2 = torch.cat((tf.logical_not(indices), indices), dim=0)
        mix_group_1 = mix_group_1.cpu().numpy()
        mix_group_2 = mix_group_2.cpu().numpy()

        dist_g1 = mat_slice(pair_dist, mix_group_1)
        dist_g2 = mat_slice(pair_dist, mix_group_2)
        dist_g1g2 = mat_slice(pair_dist, mix_group_1, mix_group_2)

    return dist_g1, dist_g1g2, dist_g2

def rand_mmd_g_xy(dist_xx, dist_xy, dist_yy, batch_size=None, dist_yx=None, omega=0.5, max_iter=3, name='mmd',
        do_summary=False, scope_prefix=''):
    """ This function calculates the mmd between two samples x and y. It uses a global sigma to make e_k match the
        given omega which is sampled uniformly. The sigma is initialized with geometric mean of dist_xy and updated
        with Newton's method.
        :param dist_xx:
        :param dist_xy:
        :param dist_yy:
        :param dist_yx: optional, if dist_xy and dist_yx are not the same
        :param batch_size: do not provide batch_size when the diagonal part of k** also need to be considered.
        :param omega:
        :param max_iter:
        :param name:
        :param do_summary:
        :param scope_prefix:
        :return:
        """

    def kernel(dist, b):
        return torch.exp(-dist * b)

    def f(b):
        k = kernel(dist_xy, b)
        e_k = torch.mean(k)
        return e_k - omega, k

    def df(k):
        kd = -k * dist_xy  # gradient of exp(-d*w)
        e_kd = torch.mean(kd)
        return e_kd

    def f_plus(b):
        k0 = kernel(dist_xy, b)
        e_k0 = torch.mean(k0)
        k1 = kernel(dist_yx, b)
        e_k1 = torch.mean(k1)
        return e_k0 + e_k1 - 2.0 * omega, (k0, k1)

    def df_plus(k):
        kd0 = -k[0] * dist_xy  # gradient of exp(-d*w)
        kd1 = -k[1] * dist_yx  # gradient of exp(-d*w)
        e_kd = torch.mean(kd0) + torch.mean(kd1)
        return e_kd

    if dist_yx is None:
        # torch.log()
        # initialize sigma as the geometric mean of dist_xy
        if torch.cuda.is_available():
            # device = dist_xy.get_device()
            beta = -torch.log(torch.tensor(omega).cuda()) / torch.mean(dist_xy + EPSI)  # beta = 1/2/sigma
        else:
            beta = -torch.log(torch.tensor(omega)) / torch.mean(dist_xy + EPSI)  # beta = 1/2/sigma
        # if max_iter is larger than one, do newton's update
        if max_iter > 1:
            for i in range(max_iter):
                beta,_ = newton_root(beta,f,df,step=i)

    else:
        # initialize sigma as the geometric mean of dist_xy and dist_yx
        # beta = 1/2/sigma
        if torch.cuda.is_available():
            # device = dist_xy.get_device()
            beta = -2.0 * torch.log(torch.tensor(omega).cuda()) / (torch.mean(dist_xy) + torch.mean(dist_yx) + EPSI)
        else:
            beta = -2.0 * torch.log(torch.tensor(omega)) / (torch.mean(dist_xy) + torch.mean(dist_yx) + EPSI)
        # if max_iter is larger than one, do newton's update
        if max_iter > 1:
            for i in range(max_iter):
                beta,_ = newton_root(beta,f_plus,df_plus,step=i)

    k_xx = kernel(dist_xx, beta)
    k_xy = kernel(dist_xy, beta)
    k_yy = kernel(dist_yy, beta)

    if batch_size is None:  # include diagonal elements in k**
        e_kxx = torch.mean(k_xx)
        e_kxy = torch.mean(k_xy)
        e_kyy = torch.mean(k_yy)
    else:  # exclude diagonal elements in k**
        # m = tf.constant(batch_size, tf.float32)
        m = batch_size
        e_kxx = matrix_mean_wo_diagonal(k_xx, m)
        e_kxy = matrix_mean_wo_diagonal(k_xy, m)
        e_kyy = matrix_mean_wo_diagonal(k_yy, m)

    if dist_yx is None:
        return e_kxx + e_kyy - 2.0 * e_kxy
    else:
        k_yx = kernel(dist_yx, beta)
        if batch_size is None:
            e_kyx = torch.mean(k_yx)
        else:
            # m = tf.constant(batch_size, tf.float32)
            m = batch_size
            e_kyx = matrix_mean_wo_diagonal(k_yx, m)
        return e_kxx + e_kyy - e_kxy - e_kyx

def rand_mmd_g_xy_bounded(
        dist_xx, dist_xy, dist_yy, batch_size=None, dist_yx=None, omega=0.5, max_iter=3, name='mmd',
        beta_lb=0.125, beta_ub=2.0, do_summary=False, scope_prefix=''):
    """ This function calculates the mmd between two samples x and y. It uses a global sigma to make e_k match the
    given omega which is sampled uniformly. The sigma is initialized with geometric mean of dist_xy and updated
    with Newton's method.
    :param dist_xx:
    :param dist_xy:
    :param dist_yy:
    :param dist_yx: optional, if dist_xy and dist_yx are not the same
    :param batch_size: do not provide batch_size when the diagonal part of k** also need to be considered.
    :param omega:
    :param max_iter:
    :param name:
    :param beta_lb: lower bound for beta (upper bound for sigma)
    :param beta_ub: upper bound for beta (lower bound for sigma)
    :param do_summary:
    :param scope_prefix:
    :return:
    """

    def kernel(dist, b):
        return torch.exp(-dist * b)

    def f(b):
        k = kernel(dist_xy, b)
        e_k = torch.mean(k)
        return e_k - omega, k

    def df(k):
        kd = -k * dist_xy  # gradient of exp(-d*w)
        e_kd = torch.mean(kd)
        return e_kd

    def f_plus(b):
        k0 = kernel(dist_xy, b)
        e_k0 = torch.mean(k0)
        k1 = kernel(dist_yx, b)
        e_k1 = torch.mean(k1)
        return e_k0 + e_k1 - 2.0 * omega, (k0, k1)

    def df_plus(k):
        kd0 = -k[0] * dist_xy  # gradient of exp(-d*w)
        kd1 = -k[1] * dist_yx  # gradient of exp(-d*w)
        e_kd = torch.mean(kd0) + torch.mean(kd1)
        return e_kd

    if dist_yx is None:
        # initialize sigma as the geometric mean of dist_xy
        if torch.cuda.is_available():
            # device = dist_xy.get_device()
            beta = -torch.log(torch.tensor(omega).cuda()) / torch.mean(dist_xy + EPSI)  # beta = 1/2/sigma
        else:
            beta = -torch.log(torch.tensor(omega)) / torch.mean(dist_xy + EPSI)  # beta = 1/2/sigma
        # if max_iter is larger than one, do newton's update
        if max_iter > 0:
            for i in range(max_iter):
                beta,_ = newton_root(beta,f,df,step=i)
            # beta, _ = tf.while_loop(
            #     cond=lambda _1, i: i < max_iter,
            #     body=lambda b, i: newton_root(b, f, df, step=i),
            #     loop_vars=(beta, tf.constant(0, dtype=tf.int32)))
    else:
        # initialize sigma as the geometric mean of dist_xy and dist_yx
        # beta = 1/2/sigma
        if torch.cuda.is_available():
            # device = dist_xy.get_device()
            beta = -2.0 * torch.log(torch.tensor(omega).cuda()) / (torch.mean(dist_xy) + torch.mean(dist_yx) + EPSI)
        else:
            beta = -2.0 * torch.log(torch.tensor(omega)) / (torch.mean(dist_xy) + torch.mean(dist_yx) + EPSI)
        # if max_iter is larger than one, do newton's update
        if max_iter > 0:
            for i in range(max_iter):
                beta,_ = newton_root(beta,f_plus,df_plus,step=i)
            # beta, _ = tf.while_loop(
            #     cond=lambda _1, i: i < max_iter,
            #     body=lambda b, i: newton_root(b, f_plus, df_plus, step=i),
            #     loop_vars=(beta, tf.constant(0, dtype=tf.int32)))

    beta = torch.clamp(beta, beta_lb, beta_ub)
    k_xx = kernel(dist_xx, beta)
    k_xy = kernel(dist_xy, beta)
    k_yy = kernel(dist_yy, beta)
    k_xx_b = kernel(torch.max(dist_xx, 0.125 / beta), beta)
    k_xy_b = kernel(torch.max(dist_xy, 2.0 / beta), beta)
    k_yy_b = kernel(torch.max(dist_yy, 0.125 / beta), beta)

    if batch_size is None:  # include diagonal elements in k**
        e_kxx = torch.mean(k_xx)
        e_kxy = torch.mean(k_xy)
        e_kyy = torch.mean(k_yy)
        e_kxx_b = torch.mean(k_xx_b)
        e_kxy_b = torch.mean(k_xy_b)
        e_kyy_b = torch.mean(k_yy_b)
    else:  # exclude diagonal elements in k**
            # m = tf.constant(batch_size, tf.float32)
        m = batch_size
        e_kxx = matrix_mean_wo_diagonal(k_xx, m)
        e_kxy = matrix_mean_wo_diagonal(k_xy, m)
        e_kyy = matrix_mean_wo_diagonal(k_yy, m)
        e_kxx_b = matrix_mean_wo_diagonal(k_xx_b, m)
        e_kxy_b = matrix_mean_wo_diagonal(k_xy_b, m)
        e_kyy_b = matrix_mean_wo_diagonal(k_yy_b, m)

    if dist_yx is None:
        return e_kxx + e_kyy - 2.0 * e_kxy, e_kxx_b - 2.0 * e_kyy_b + e_kxy_b
    else:
        k_yx = kernel(dist_yx, beta)
        # k_yx_b = kernel(tf.minimum(dist_yx, upper_bound), beta)
        if batch_size is None:
            e_kyx = torch.mean(k_yx)
            # e_kyx_b = tf.reduce_mean(k_yx_b)
        else:
            m = batch_size
            e_kyx = matrix_mean_wo_diagonal(k_yx, m)
                # e_kyx_b = matrix_mean_wo_diagonal(k_yx_b, m)
            # if do_summary:
            #     with tf.name_scope(None):  # return to root scope to avoid scope overlap
            #         tf.summary.scalar(scope_prefix + name + 'kyx', e_kyx)
            #         # tf.summary.scalar(scope_prefix + name + 'kyx_b', e_kyx_b)
        return e_kxx + e_kyy - e_kxy - e_kyx

def rand_mmd_g_xn(
        x, y_rho, batch_size, d, y_mu=0.0, dist_xx=None, omega=0.5, max_iter=0, name='mmd',
        do_summary=False, scope_prefix=''):
    """ This function calculates the mmd between two samples x and y. y is sampled from normal distribution
    with zero mean and specified STD. This function uses a global sigma to make e_k match the given omega
    which is sampled uniformly. The sigma is initialized with geometric mean of dist_xy and updated with
    Newton's method.
    :param x:
    :param y_rho: y_std = sqrt(y_rho / 2.0 / d)
    :param batch_size:
    :param d: number of features in x
    :param y_mu:
    :param dist_xx:
    :param omega:
    :param max_iter:
    :param name:
    :param do_summary:
    :param scope_prefix:
    :return:
    """
    if dist_xx is None:
        xxt = torch.matmul(x, x.t())
        dx = torch.diag(xxt)
        if torch.cuda.is_available():
            # device = dx.get_device()
            dist_xx = torch.max(torch.unsqueeze(dx, dim=1) - 2.0 * xxt + torch.unsqueeze(dx, dim=0), torch.Tensor([0.0]).cuda())
        else:
            dist_xx = torch.max(torch.unsqueeze(dx, dim=1) - 2.0 * xxt + torch.unsqueeze(dx, dim=0), torch.Tensor([0.0]))
        # get dist(x, Ey)
    dist_xy = torch.sum(torch.mul(x - y_mu, x - y_mu), dim=1)

    def kernel(dist, b):
        return torch.exp(-dist * b)

    def f(b):
        const_f = d / (d + b * y_rho)
        if torch.cuda.is_available():
            # device = dist_xy.get_device()
            k = torch.pow(const_f, torch.tensor(d / 2.0).cuda()) * torch.exp(-b * const_f * dist_xy)
        else:
            k = torch.pow(const_f, torch.tensor(d / 2.0)) * torch.exp(-b * const_f * dist_xy)
        e_k = torch.mean(k)
        return e_k - omega, (const_f, k, e_k)

    def df(k):

        if torch.cuda.is_available():
            # device = dist_xy.get_device()
            kd = -y_rho * k[0] / 2.0 * k[2] - torch.mean(torch.pow(k[0], torch.tensor(2).cuda()) * dist_xy * k[1])  # gradient of exp(-d*w)
        else:
            kd = -y_rho * k[0] / 2.0 * k[2] - torch.mean(
                torch.pow(k[0], torch.tensor(2)) * dist_xy * k[1])  # gradient of exp(-d*w)
        e_kd = torch.mean(kd)
        return e_kd

        # initialize sigma as the geometric mean of dist_xy
    if torch.cuda.is_available():
        # device = dist_xy.get_device()
        beta = -torch.log(torch.tensor(omega).cuda()) / (torch.mean(dist_xy) + y_rho / 2.0)  # beta = 1/2/sigma
    else:
        beta = -torch.log(torch.tensor(omega)) / (torch.mean(dist_xy) + y_rho / 2.0)  # beta = 1/2/sigma
    # if max_iter is larger than one, do newton's update
    if max_iter > 0:
        beta, _ = tf.while_loop(
                cond=lambda _1, i: i < max_iter,
                body=lambda b, i: newton_root(b, f, df, step=i),
                loop_vars=(beta, tf.constant(0, dtype=tf.int32)))

    const_0 = d / (d + beta * y_rho)
    k_xx = kernel(dist_xx, beta)
    if torch.cuda.is_available():
        # device = dist_xy.get_device()
        k_xy = torch.pow(const_0, torch.tensor(d / 2.0).cuda()) * torch.exp(-beta * const_0 * dist_xy)
    else:
        k_xy = torch.pow(const_0, torch.tensor(d / 2.0)) * torch.exp(-beta * const_0 * dist_xy)
    e_kxx = matrix_mean_wo_diagonal(k_xx, batch_size)
    e_kxy = torch.mean(k_xy)

    if torch.cuda.is_available():
        # device = e_kxy.get_device()
        e_kyy = torch.pow(d / (d + 2.0 * beta * y_rho), torch.tensor(d / 2.0).cuda())
    else:
        e_kyy = torch.pow(d / (d + 2.0 * beta * y_rho), torch.tensor(d / 2.0))
    return e_kxx + e_kyy - 2.0 * e_kxy

def newton_root(x, f, df, step=None):
    """ This function does one iteration update on x to find the root f(x)=0. It is primarily used as the body of
    tf.while_loop.
    :param x:
    :param f: a function that receives x as input and outputs f(x) and other info for gradient calculation
    :param df: a function that receives info as inputs and outputs the gradient of f at x
    :param step:
    :return:
    """
    fx, info2grad = f(x)
    gx = df(info2grad)
    x = x - fx / (gx + EPSI)

    if step is None:
        return x
    else:
        return x, step + 1

def mmd_rand_g_loss(s_recon,s_x,omega_range = [0.05, 0.85],ref_normal = 1.0,num_scores=None):
    """ maximum mean discrepancy with gaussian kernel and random kernel scale
    """
    # calculate pairwise distance
    dist_gg, dist_gd, dist_dd = get_squared_dist(
        s_recon, s_x, z_score=False)
    omega_raw = random.uniform(omega_range[0],omega_range[1])
    omega = torch.tensor(omega_raw) \
        if isinstance(omega_range, (list, tuple)) else omega_range

    batch_size = s_recon.size()[0]

    loss_gr = rand_mmd_g_xy(
        dist_gg, dist_gd, dist_dd, batch_size, omega=omega_raw,
        max_iter=3, name='mmd_gr', scope_prefix='rand_g/')
    loss_gn = rand_mmd_g_xn(
        s_recon, ref_normal, batch_size, num_scores, dist_xx=dist_gg, omega=omega_raw,
        max_iter=3, name='mmd_gn', scope_prefix='rand_g/')
    loss_rn = rand_mmd_g_xn(
        s_x, ref_normal, batch_size, num_scores, dist_xx=dist_dd, omega=omega_raw,
        max_iter=3, name='mmd_rn',scope_prefix='rand_g/')
    # final loss
    loss_gen = loss_gr
    loss_dis = loss_rn - loss_gr
    return loss_dis,loss_gen

# torch.nn.ReLU()
def mmd_rand_g_bounded_loss(s_recon,s_x,omega_range = [0.05, 0.85],ref_normal = 1.0,num_scores=None,beta_lb=0.125, beta_ub=2.0):
    """ maximum mean discrepancy with gaussian kernel and random kernel scale, and upper bounds on dxy
            :return:
            """
    # calculate pairwise distance
    dist_gg, dist_gd, dist_dd = get_squared_dist(s_recon,s_x, z_score=False)
    omega_raw = random.uniform(omega_range[0], omega_range[1])
    batch_size = s_recon.size()[0]
    loss_gr, loss_gr_b = rand_mmd_g_xy_bounded(dist_gg, dist_gd, dist_dd,batch_size,omega=omega_raw,beta_lb=beta_lb,beta_ub=beta_ub,
                                               max_iter=3, name='mmd',scope_prefix='rand_g/')
    loss_gen = loss_gr
    loss_dis = - loss_gr_b
    return loss_dis,loss_gen

def mmd_sym_rand_g_loss(s_recon,s_x,omega_range = [0.05, 0.85],ref_normal = 1.0,num_scores=None):
    """ Version 2 of symmetric rand_g. This function does not use label smoothing
           This function does not work.
           :return:
           """
    # calculate pairwise distance
    batch_size = s_recon.size()[0]
    pair_dist = get_squared_dist(tf.concat((s_recon, s_x), axis=0))
    dist_gg, dist_gd, dist_dd = slice_pairwise_distance(pair_dist, batch_size=batch_size)
    # mmd
    # omega = tf.random_uniform([], self.omega_range[0], self.omega_range[1], dtype=tf.float32) \
    #     if isinstance(self.omega_range, (list, tuple)) else self.omega_range
    omega_raw = random.uniform(omega_range[0],omega_range[1])
    loss_gr = rand_mmd_g_xy(
        dist_gg, dist_gd, dist_dd, batch_size, omega=omega_raw,
        max_iter=3, name='mmd_gr',scope_prefix='sym_rg_mix/')
    loss_gn = rand_mmd_g_xn(
        s_recon, ref_normal, batch_size, num_scores, y_mu=-0.5, dist_xx=dist_gg,
        omega=omega_raw, max_iter=3, name='mmd_gn',scope_prefix='sym_rg_mix/')
    loss_rn = rand_mmd_g_xn(
        s_recon, ref_normal,batch_size, num_scores, y_mu=0.5, dist_xx=dist_dd,
        omega=omega_raw, max_iter=3, name='mmd_rn', scope_prefix='sym_rg_mix/')
    loss_gen = loss_gr
    loss_dis = 0.5 * (loss_rn + loss_gn) - loss_gr

    return loss_dis,loss_gen

def mmd_repulsive_g_loss(s_recon,s_x,repulsive_weights = [0.0, -1.0],dis_penalty=None,dis_scale=None):
        """ repulsive loss
        :return:
        """
        # calculate pairwise distance
        dist_gg, dist_gd, dist_dd = get_squared_dist(
            s_recon, s_x, z_score=False)
        batch_size = s_recon.size()[0]
        # self.loss_gen, self.loss_dis = mmd_g(
        #     dist_gg, dist_gd, dist_dd, self.batch_size, sigma=1.6,
        #     name='mmd_g', do_summary=self.do_summary, scope_prefix='', custom_weights=self.repulsive_weights)
        loss_gen, loss_dis = mmd_g(
            dist_gg, dist_gd, dist_dd, batch_size, sigma=1.0,
            name='mmd_g',scope_prefix='', custom_weights=repulsive_weights)
        if dis_penalty is not None:
            loss_dis = loss_dis + dis_penalty
        if dis_scale is not None:
            loss_dis = (loss_dis - 1.0) * dis_scale
        return loss_dis,loss_gen

def mmd_repulsive_g_bounded_loss(s_recon,s_x,repulsive_weights = [0.0, -1.0],dis_penalty=None,dis_scale=None):
        """ rmb loss
        :return:
        """
        # calculate pairwise distance
        batch_size = s_recon.size()[0]
        dist_gg, dist_gd, dist_dd = get_squared_dist(
            s_recon, s_x, z_score=False)
        loss_gen, loss_dis = mmd_g_bounded(
            dist_gg, dist_gd, dist_dd, batch_size, sigma=1.0, lower_bound=0.25, upper_bound=4.0,
            name='mmd_g',scope_prefix='', custom_weights=repulsive_weights)
        if dis_penalty is not None:
            loss_dis = loss_dis + dis_penalty
        if dis_scale is not None:
            loss_dis = loss_dis * dis_scale

        return loss_dis,loss_gen

class Mis_losses(object):
    def __init__(self,types='mix_g'):
        self.loss_average = None
        # self.loss = None
        self.types= types
        self.tensor =None
        self.sig = None

    def moving_average_update(self,name, shape, tensor_update, rho=0.01, clip_values=None,
                              dtype=tf.float32):
        """ This function creates a tensor that will be updated by tensor_update using moving average
        :param tensor_update: update at each iteration
        :param name: name for the tensor
        :param shape: shape of tensor
        :param rho:
        :param initializer:
        :param clip_values:
        :param dtype:
        :return:
        """
        if self.tensor is None:
            if torch.cuda.is_available():
                try:
                    self.tensor =  torch.tensor(0).cuda()
                except Exception as e:
                    self.tensor = torch.tensor(0).cuda(1)
            else:
                self.tensor =  torch.tensor(0)
        # tensor = tf.get_variable(
        #     name, shape=shape, dtype=dtype, initializer=initializer, trainable=False)
        if clip_values is None:
            # tf.add_to_collection(
            #     tf.GraphKeys.UPDATE_OPS,
            #     tf.assign(tensor, tensor + rho * tensor_update))
            self.tensor = tensor_update*rho+self.tensor
        else:
            self.tensor = torch.clamp(self.tensor,min=clip_values[0],max=clip_values[1])
            # tf.add_to_collection(
            #     tf.GraphKeys.UPDATE_OPS,
            #     tf.assign(
            #         tensor,
            #         tf.clip_by_value(
            #             tensor + rho * tensor_update,
            #             clip_value_min=clip_values[0], clip_value_max=clip_values[1])))

        # return tensor

    def moving_average_copy(self,tensor, name=None, rho=0.01):
        tensor_copy = tensor.clone().detach()
        tensor_copy.requires_grad = False
        tensor_copy = tensor_copy * (1 - rho) + tensor
        self.loss_average = tensor_copy

    def get_loss_average(self):
        return self.loss_average

    def update_moving_average_copy(self,tensor,rho=0.01):
        self.loss_average = self.loss_average * (1 - rho) + tensor * rho
        self.loss_average.requires_grad = False

    def get_mix_coin(self,
        loss, loss_threshold, batch_size=None, loss_average_update=0.01,
                mix_prob_update=0.01,
        loss_average_name='loss_ave'):
            """ This function generate a mix_indices to mix data from two classes
            :param loss:
            :param loss_threshold:
            :param batch_size:
            :param loss_average_update:
            :param mix_prob_update:
            :param loss_average_name:
            :return:
            """
            # calculate moving average of loss
            if self.loss_average == None:
                self.moving_average_copy(loss, loss_average_name, rho=loss_average_update)
            else:
                self.update_moving_average_copy(loss,rho=loss_average_update)
            self.moving_average_update('prob', [], self.loss_average - loss_threshold)
            # sample mix_indices
            uni = torch.rand([batch_size])
            torch.gt()
            mix_indices = tf.greater(uni, self.tensor, name='mix_indices')  # mix_indices for using original data
            return mix_indices


    def mmd_g_mix_loss(self,s_recon, s_x, mix_threshold=1.0, dis_penalty=None, sigma=[1, np.sqrt(2), 2, np.sqrt(8), 4]):
        """ maximum mean discrepancy with gaussian kernel and mixing score_gen and score_data
                    if discriminator is too strong
                    :param mix_threshold:
                    :return:
                    """
        # calculate pairwise distance
        pair_dist = get_squared_dist(torch.cat((s_recon, s_x), dim=0))
        batch_size = s_recon.size()[0]
        dist_gg, dist_gd, dist_dd = slice_pairwise_distance(pair_dist, batch_size=batch_size)

        loss_gen = mixture_mmd_g(dist_gg, dist_gd, dist_dd, batch_size, sigma=sigma,
                                     name='mmd', scope_prefix='mmd_g_mix/')
        # mix data if self.loss_gen surpass loss_gen_threshold
        mix_indices = self.get_mix_coin(loss_gen,mix_threshold,batch_size,loss_average_name='gen_average')
        dist_gg_mix, dist_gd_mix, dist_dd_mix = slice_pairwise_distance(pair_dist,batch_size=batch_size,indices=mix_indices)
        loss_mix = mixture_mmd_g(
                dist_gg_mix, dist_gd_mix, dist_dd_mix, batch_size, sigma=sigma,
                name='mmd_mix', scope_prefix='mmd_g_mix/')
        loss_dis = -loss_mix
        return loss_dis,loss_gen

    def _single_mmd_g_mix_(self,s_recon, s_x,mix_threshold=0.2):
        """ maximum mean discrepancy with gaussian kernel and mixing score_gen and score_data
               if discriminator is too strong
               :param mix_threshold:
               :return:
               """
        # calculate pairwise distance
        # calculate pairwise distance
        batch_size = s_recon.size()[0]
        pair_dist = get_squared_dist(torch.cat((s_recon, s_x), dim=0))
        dist_gg, dist_gd, dist_dd = slice_pairwise_distance(pair_dist,batch_size=batch_size)
        # loss_gen = mixture_mmd_g(dist_gg, dist_gd, dist_dd, batch_size, sigma=1.0,
        #                          name='mmd', scope_prefix='mmd_g_mix/')
        # # mix data if self.loss_gen surpass loss_gen_threshold
        # mix_indices = self.get_mix_coin(loss_gen, mix_threshold, batch_size, loss_average_name='gen_average')
        loss_gen = mmd_g(dist_gg, dist_gd, dist_dd,batch_size=batch_size,sigma=1.0,
                         name='mmd',scope_prefix='mmd_g_mix/')
        # mix data if self.loss_gen surpass loss_gen_threshold
        mix_indices = self.get_mix_coin(
            loss_gen, mix_threshold, batch_size=batch_size, loss_average_name='gen_average')
        dist_gg_mix, dist_gd_mix, dist_dd_mix = slice_pairwise_distance(pair_dist,batch_size=batch_size,indices=mix_indices)
        # mmd for mixed data
        loss_mix = mmd_g(
            dist_gg_mix, dist_gd_mix, dist_dd_mix, batch_size, sigma=1.0,
            name='mmd_mix', scope_prefix='mmd_g_mix/')
        loss_dis = -loss_mix

        return loss_dis,loss_gen

    def _rand_g_mix_(self,s_recon, s_x,mix_threshold=0.2,omega_range = [0.05, 0.85],ref_normal = 1.0,num_scores=None,beta_lb=0.125, beta_ub=2.0):
        """ maximum mean discrepancy with gaussian kernel and random kernel scale
                and mixing score_gen and score_data if discriminator is too strong
                """
        # calculate pairwise distance
        batch_size = s_recon.size()[0]
        pair_dist = get_squared_dist(torch.cat((s_recon, s_x),dim=0))
        dist_gg, dist_gd, dist_dd = slice_pairwise_distance(pair_dist,batch_size)
        omega_raw = random.uniform(omega_range[0],omega_range[1])
        loss_gr = rand_mmd_g_xy(
            dist_gg, dist_gd, dist_dd,batch_size, omega=omega_raw,
            max_iter=3, name='mmd_gr', scope_prefix='rand_g_mix/')
        loss_gn = rand_mmd_g_xn(
            s_recon, ref_normal, batch_size, num_scores, dist_xx=dist_gg, omega=omega_raw,
            max_iter=3, name='mmd_gn',scope_prefix='rand_g_mix/')
        loss_rn = rand_mmd_g_xn(
            s_recon, ref_normal,batch_size, num_scores, dist_xx=dist_dd, omega=omega_raw,
            max_iter=3, name='mmd_rn',scope_prefix='rand_g_mix/')

        # mix data if self.loss_gen surpass loss_gen_threshold
        mix_indices = self.get_mix_coin(loss_gr,mix_threshold,batch_size)
        dist_gg_mix, dist_gd_mix, dist_dd_mix = slice_pairwise_distance(pair_dist, indices=mix_indices)
        # mmd for mixed data
        loss_gr_mix = rand_mmd_g_xy(
            dist_gg_mix, dist_gd_mix, dist_dd_mix, batch_size, omega=omega_raw,
            max_iter=3, name='mmd_gr_mix',scope_prefix='rand_g_mix/')
        # final loss
        loss_gen = loss_gr
        loss_dis = loss_rn - loss_gr_mix

        return loss_dis,loss_gen

    def _sym_rg_mix_(self,s_recon,s_x,mix_threshold=0.2,omega_range = [0.05, 0.85],ref_normal = 1.0,num_scores=None):
        """ symmetric version of rand_g_mix
               :param mix_threshold:
               :return:
        """
        batch_size = s_recon.size()[0]
        # calculate pairwise distance
        pair_dist = get_squared_dist(torch.cat((s_recon, s_x), dim=0))
        dist_gg, dist_gd, dist_dd = slice_pairwise_distance(pair_dist, batch_size=batch_size)
        # mmd
        omega_raw = random.uniform(omega_range[0],omega_range[1])
        loss_gr = rand_mmd_g_xy(
            dist_gg, dist_gd, dist_dd,batch_size, omega=omega_raw,
            max_iter=3, name='mmd_gr',scope_prefix='sym_rg_mix/')
        loss_gn = rand_mmd_g_xn(
            s_recon, ref_normal, batch_size, num_scores, dist_xx=dist_gg, omega=omega_raw,
            max_iter=3, name='mmd_gn',scope_prefix='sym_rg_mix/')
        loss_rn = rand_mmd_g_xn(
            s_x, ref_normal, batch_size, num_scores, dist_xx=dist_dd, omega=omega_raw,
            max_iter=3, name='mmd_rn',scope_prefix='sym_rg_mix/')

        mix_indices = self.get_mix_coin(loss_gr, mix_threshold, batch_size=batch_size, loss_average_name='gr_average')
        dist_gg_mix, dist_gd_mix, dist_dd_mix = slice_pairwise_distance(pair_dist, indices=mix_indices)
        # mmd for mixed data
        loss_gr_mix = rand_mmd_g_xy(
            dist_gg_mix, dist_gd_mix, dist_dd_mix,batch_size, omega=omega_raw,
            max_iter=3, name='mmd_gr_mix',scope_prefix='sym_rg_mix/')

        # final loss
        loss_gen = loss_gr + loss_gn
        loss_dis = loss_rn - loss_gr_mix - loss_gn
        return loss_dis,loss_gen

    def _rand_g_instance_noise_(self,s_recon, s_x, mix_threshold=0.2,omega_range = [0.05, 0.85],ref_normal = 1.0,num_scores=None):
        """ This function tests instance noise
                :param mix_threshold:
                :return:
                """
        if self.sig is None:
            if torch.cuda.is_available():
                # device = s_recon.get_device()
                self.sig = torch.tensor(0).cuda()
            else:
                self.sig = torch.tensor(0)
        # sigma =
        stddev = torch.log(self.sig + 1.0)
        # noise_gen = torch.randn(list(s_recon.size()))
        # noise_x = torch.randn(list(s_x.size()))
    #
        noise_gen = torch.normal(mean=0.0,std=stddev.item(),size=list(s_recon.size()))
        noise_x = torch.normal(mean=0.0, std=stddev.item(), size=list(s_x.size()))
        score_gen = s_recon + noise_gen
        score_data = s_x + noise_x

        loss_dis,loss_gen = mmd_rand_g_loss(score_gen,score_data,omega_range,ref_normal=ref_normal,num_scores=num_scores)
        if self.loss_average is None:
            self.moving_average_copy(loss_gen,'mmd_mean')
        else:
            self.update_moving_average_copy(loss_gen)
        self.sig = torch.clamp(self.sig+ 0.001 * (self.loss_average - mix_threshold),min=0.0,max=1.7183)

        return loss_dis,loss_gen

    # def __call__(self, s_recon,s_x):
    #     if self.types







    # def __call__(self, *args, **kwargs):







