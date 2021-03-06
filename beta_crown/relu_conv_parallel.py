import time
import numpy as np
import torch

from .branch_and_bound import pick_out_batch, add_domain_parallel, ReLUDomain
from .babsr_score_conv import choose_node_parallel

visited = 0


def batch_verification(d, net, batch, pre_relu_indices, no_LP, growth_rate, decision_thresh=0, layer_set_bound=True, beta=True):
    global visited

    mask, orig_lbs, orig_ubs, slopes, selected_domains = pick_out_batch(d, decision_thresh, batch, device=net.x.device)
    relu_start = time.time()
    if mask is not None:
        branching_decision = choose_node_parallel(orig_lbs, orig_ubs, mask, net.layers, pre_relu_indices,
                                                  None, 0, batch=batch)

        print('splitting decisions: {}'.format(branching_decision))
        # print('history', selected_domains[0].history)
        history = [sd.history for sd in selected_domains]
        # print(len(history), history, len(branching_decision))

        ret = net.get_lower_bound(orig_lbs, orig_ubs, branching_decision, slopes=slopes, history=history,
                                  decision_thresh=decision_thresh, layer_set_bound=layer_set_bound, beta=beta)
        dom_ub, dom_lb, dom_ub_point, updated_mask, dom_lb_all, dom_ub_all, slopes = ret
        print('dom_lb parallel: ', dom_lb[:10])
        bd_0, bd_1 = [], []
        for bd in branching_decision:
            bd_0.append([(bd, 1)])
            bd_1.append([(bd, 0)])

        unsat_list = add_domain_parallel(updated_mask, lb=dom_lb, ub=dom_ub, lb_all=dom_lb_all, up_all=dom_ub_all,
                                         domains=d, selected_domains=selected_domains, slope=slopes,
                                         growth_rate=growth_rate, branching_decision=bd_0+bd_1, decision_thresh=decision_thresh)
        visited += (len(selected_domains) - len(unsat_list)) * 2  # one unstable neuron split to two nodes

    relu_end = time.time()
    print('relu split requires (KW): ', relu_end - relu_start)
    print('length of domains:', len(d))

    if len(d) > 0:
        global_lb = d[0].lower_bound
    else:
        print("No domains left, verification finished!")
        return 999

    print(f"Current lb:{global_lb.item()}")

    print('{} neurons visited'.format(visited))

    return global_lb


def relu_bab_parallel(net, domain, x, batch=64, no_LP=False, decision_thresh=0,
                      use_neuron_set_strategy=False, beta=True, max_subproblems_list=100000, timeout=3600):
    start = time.time()
    global visited
    visited = 0
    global_ub, global_lb, global_ub_point, duals, primals, updated_mask, lower_bounds, upper_bounds, pre_relu_indices, slope = net.build_the_model(
        domain, x, no_lp=no_LP, decision_thresh=decision_thresh)

    print(global_lb)
    if global_lb > decision_thresh:
        return global_lb, global_ub, global_ub_point, 0
    
    
    #lower_bounds = [i if i != [] else torch.tensor([]).to(net.x.device) for i in lower_bounds]
    #upper_bounds = [i if i != [] else torch.tensor([]).to(net.x.device) for i in upper_bounds]
    #print(len(lower_bounds))
    candidate_domain = ReLUDomain(updated_mask, global_lb, global_ub, lower_bounds, upper_bounds, slope, depth=0).to_cpu()
    #candidate_domain = ReLUDomain(updated_mask, global_lb, global_ub, lower_bounds, upper_bounds, slope, depth=0)
    if no_LP:
        domains = [candidate_domain]
    else:
        domains = [candidate_domain]
    tot_ambi_nodes = 0
    for layer_mask in updated_mask:
        tot_ambi_nodes += torch.sum(layer_mask == -1).item()

    print('# of unstable neurons:', tot_ambi_nodes)
    random_order = np.arange(tot_ambi_nodes)
    np.random.shuffle(random_order)

    # glb_record = [[time.time()-start, global_lb.item()]]
    while len(domains) > 0:

        if len(domains) > 80000 and len(domains) % 10000 < batch * 2 and use_neuron_set_strategy:  # do two batch of neuron set bounds  per 10000 domains
            # neuron set  bounds cost more memory, we set a smaller batch here
            global_lb = batch_verification(domains, net, int(batch/2), pre_relu_indices, no_LP, 0,
                                           decision_thresh=decision_thresh, layer_set_bound=False, beta=beta)
        else:
            global_lb = batch_verification(domains, net, batch, pre_relu_indices, no_LP, 0, decision_thresh=decision_thresh, beta=beta)
        if len(domains) > max_subproblems_list:
            print("no enough memory for the domain list")
            del domains
            return global_lb, np.inf, None, visited

        if time.time() - start > timeout:
            print('time out!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            del domains
            # np.save('glb_record.npy', np.array(glb_record))
            return global_lb, np.inf, None, visited

        # glb_record.append([time.time() - start, global_lb.item()])

    del domains
    return global_lb, np.inf, None, visited
