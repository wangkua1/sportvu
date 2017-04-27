def find_position_idx_by_name(name, event_obj):
    names = [event_obj.player_ids_dict[p.id][0] for (idx,p) in enumerate(event_obj.moments[0].players)]    
    for i, n in enumerate(names):
        if n == name:
            return i
    return -1