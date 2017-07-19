from __future__ import division
from Constant import Constant
from Moment import Moment
from Team import Team
plt = None
from sportvu import data
import numpy as np

class EventException(Exception):
    pass

def format_pbp(pbp):
    event_str = "Play-By-Play Annotations\n"
    g = pbp.iterrows()
    for eind, pp in pbp.iterrows():
      event_str += '------------Event: %i ---------------\n' % eind
      event_str += str(pp['HOMEDESCRIPTION'])+ " , " +\
              str(pp['VISITORDESCRIPTION'])+ " , "+\
              str(pp['PCTIMESTRING'])+ " , "+\
              str(pp['event_str'])  + '\n'
    return event_str

def get_position_number(position, add_noise=True):
    if add_noise:
        noise = np.random.rand()*0.2-0.1
    else:
        noise = 0.0
    if position == 'G':
        return 1.+noise
    if position == 'F-G':
        return 2.+noise
    if position == 'F':
        return 3.+noise
    if position == 'C-F':
        return 4.+noise
    if position == 'C':
        return 5.+noise

class Event:
    """A class for handling and showing events"""

    def __init__(self, event, gameid = ''):
        self.gameid = gameid
        moments = event['moments']
        self.moments = [Moment(moment) for moment in moments]
        if gameid != '':
          self._resolve_home_basket()
        self.home_team_id = event['home']['teamid']
        home_players = event['home']['players']
        guest_players = event['visitor']['players']
        players = home_players + guest_players
        player_ids = [player['playerid'] for player in players]
        player_names = [" ".join([player['firstname'],
                        player['lastname']]) for player in players]
        player_jerseys = [player['jersey'] for player in players]
        player_position = [player['position'] for player in players]
        player_position_numer = [get_position_number(player['position']) for player in players]
        values = list(zip(player_names, player_jerseys, player_position, player_position_numer))
        # Example: 101108: ['Chris Paul', '3', 'G']
        self.player_ids_dict = dict(zip(player_ids, values))
        self.pbp = event['playbyplay']
    def _resolve_home_basket(self):
        """
        hardcoded for the 3 games labelled
        '0021500357' q1 home: 0
        '0021500150' q1 home: 1
        '0021500278' q1 home: 0
        """
        hard_code = {'0021500357':0, '0021500150':1, '0021500278':0}
        self.home_basket = (hard_code[self.gameid] + (self.moments[0].quarter > 2))%2

    def is_home_possession(self, moment):
        ball_basket = int(moment.ball.x > 50)
        if ball_basket == self.home_basket: # HOME possession
          return True
        else: # VISITOR possession 
          return False
    def truncate_by_following_event(self, event2):
        """
        use the given event to truncate the current  (i.e. do not include the 
        trailing frames shown in a later event)
        """
        # trunctate
        end_time_from_e2 = event2['moments'][0][2]
        last_idx = -1 
        for idx, moment in enumerate(self.moments):
          if moment.game_clock < end_time_from_e2:
            last_idx = idx
            break
        if last_idx != -1:
          self.moments = self.moments[:last_idx]
    def sequence_around_t(self, T_a, tfr, seek_last=True):
        """
        segment [T_a - tfr, T_a + tfr]
        note: when seek_last = True, seek for the last T_a 
              (this detail becomes important when game-clock stops within one Event) 
        """
        T_a_index = -1
        for idx, moment in enumerate(self.moments):
          if moment.game_clock < T_a:
            T_a_index = idx
            break
        
        if T_a_index == -1: 
          # print ('EventException')
          raise EventException('bad T_a, or bad event')

        start_ind = np.max([0, T_a_index-tfr])
        end_ind = np.min([len(self.moments)-1, T_a_index + tfr])
        if end_ind - start_ind != 2*tfr:
          raise EventException('incorrect length')
        self.moments = self.moments[start_ind:end_ind]

    def update_radius(self, i, player_circles, ball_circle, annotations, clock_info, lines, pred_lines):
        line = lines[0]
        ret = [player_circles, ball_circle, line]
        if self.futures is not None and i in self.futures[0]:
          frame_ind = self.futures[0].index(i)
          for sample_idx, l in enumerate(pred_lines):
            l.set_ydata(self.futures[2][frame_ind, sample_idx,:,1])
            l.set_xdata(self.futures[2][frame_ind, sample_idx,:,0])
            ret.append(l)
          line.set_ydata(self.futures[1][frame_ind, :, 1])
          line.set_xdata(self.futures[1][frame_ind, :, 0])
          
        moment = self.moments[i]
        for j, circle in enumerate(player_circles):
            try:
              circle.center = moment.players[j].x, moment.players[j].y
            except:
              raise EventException()
            
            annotations[j].set_position(circle.center)
            clock_test = 'Quarter {:d}\n {:02d}:{:02d}\n {:03.1f}'.format(
                         moment.quarter,
                         int(moment.game_clock) % 3600 // 60,
                         int(moment.game_clock) % 60,
                         moment.shot_clock)
            clock_info.set_text(clock_test)
        ball_circle.center = moment.ball.x, moment.ball.y
        ball_circle.radius = moment.ball.radius / Constant.NORMALIZATION_COEF
        x = np.arange(Constant.X_MIN, Constant.X_MAX, 1)
        court_center_x = Constant.X_MAX /2
        court_center_y = Constant.Y_MAX /2
        player_of_interest = moment.players[7]
            
        return ret

    def show(self, save_path='', futures=None):
        global plt
        if plt is None:
          import matplotlib.pyplot as plt
        from matplotlib import animation
        from matplotlib.patches import Circle, Rectangle, Arc
        import cPickle as pkl
        # Leave some space for inbound passes
        ax = plt.axes(xlim=(Constant.X_MIN,
                            Constant.X_MAX),
                      ylim=(Constant.Y_MIN,
                            Constant.Y_MAX))
        ax.axis('off')
        fig = plt.gcf()
        ax.grid(False)  # Remove grid
        try:
          start_moment = self.moments[0]
        except IndexError as e:
          raise EventException()

        player_dict = self.player_ids_dict

        clock_info = ax.annotate('', xy=[Constant.X_CENTER, Constant.Y_CENTER],
                                 color='black', horizontalalignment='center',
                                   verticalalignment='center')

        annotations = [ax.annotate(self.player_ids_dict[player.id][1], xy=[0, 0], color='w',
                                   horizontalalignment='center',
                                   verticalalignment='center', fontweight='bold')
                       for player in start_moment.players]
        x = np.arange(Constant.X_MIN, Constant.X_MAX, 1)
        if futures is not None:
            self.futures = futures
            pred_lines = []
            for _ in range(self.futures[2].shape[1]):
              l, = ax.plot([],[], color='k', alpha=1./self.futures[2].shape[1])
              pred_lines.append(l)
        else:
            self.futures = None
            pred_lines = None
        line, = ax.plot([], [], color='w')

        # Prepare table
        sorted_players = sorted(start_moment.players, key=lambda player: player.team.id)
        
        home_player = sorted_players[0]
        guest_player = sorted_players[5]
        column_labels = tuple([home_player.team.name, guest_player.team.name])
        column_colours = tuple([home_player.team.color, guest_player.team.color])
        cell_colours = [column_colours for _ in range(5)]
        
        home_players = [' #'.join([player_dict[player.id][0], player_dict[player.id][1]]) for player in sorted_players[:5]]
        guest_players = [' #'.join([player_dict[player.id][0], player_dict[player.id][1]]) for player in sorted_players[5:]]
        players_data = list(zip(home_players, guest_players))

        try:
          table = plt.table(cellText=players_data,
                              colLabels=column_labels,
                              colColours=column_colours,
                              colWidths=[Constant.COL_WIDTH, Constant.COL_WIDTH],
                              loc='bottom',
                              cellColours=cell_colours,
                              fontsize=Constant.FONTSIZE,
                              cellLoc='center')
        except ValueError as e:
          raise EventException() ### unknown error, probably malformed sequence
        else:
          pass
        finally:
          pass
        table.scale(1, Constant.SCALE)
        table_cells = table.properties()['child_artists']
        for cell in table_cells:
            cell._text.set_color('white')

        player_circles = [plt.Circle((0, 0), Constant.PLAYER_CIRCLE_SIZE, color=player.color)
                          for player in start_moment.players]
        ball_circle = plt.Circle((0, 0), Constant.PLAYER_CIRCLE_SIZE,
                                 color=start_moment.ball.color)
        for circle in player_circles:
            ax.add_patch(circle)
        ax.add_patch(ball_circle)

        anim = animation.FuncAnimation(
                         fig, self.update_radius,
                         fargs=(player_circles, ball_circle, annotations, clock_info, [line], pred_lines),
                         frames=len(self.moments), interval=Constant.INTERVAL)
        court = plt.imread(data.constant.court_path)
        plt.imshow(court, zorder=0, extent=[Constant.X_MIN, Constant.X_MAX - Constant.DIFF,
                                            Constant.Y_MAX, Constant.Y_MIN])
        plt.title(format_pbp(self.pbp))
        if save_path == '':
          plt.show()
        else:
          plt.ioff()
          Writer = animation.writers['ffmpeg']
          writer = Writer(fps=25, metadata=dict(artist='Me'), bitrate=1800)
          anim.save(save_path, writer)
        plt.clf()