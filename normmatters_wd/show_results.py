import pickle
import matplotlib.pyplot as plt
import matplotlib.widgets as wdgt
import matplotlib
import argparse
import numpy as np
import mpld3


def autoscale_axis_aux(ax, ytop_min=None, ybottom_max=None, xleft_max=None, xright_min=None, xmargin=None):
    ax.autoscale(enable=True, axis='both', tight=True)
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    if ytop_min is None:
        ytop_min = ymax
    if ybottom_max is None:
        ybottom_max = ymin
    if xleft_max is None:
        xleft_max = xmin
    if xright_min is None:
        xright_min = xmax

    ymin = np.min([ymin, ybottom_max])
    ymax = np.max([ymax, ytop_min])
    xmin = np.min([xmin, xleft_max])
    xmax = np.max([xmax, xright_min])
    if xmargin:
        xmin -= float(np.abs(xmax-xmin)*xmargin)
        xmax += float(
            np.abs(xmax-xmin)*xmargin)

    ax.set_ylim(ymin=ymin, ymax=ymax)
    ax.set_xlim(xmin=xmin, xmax=xmax)


class ResultsPlotter:
    def __init__(self):
        self.results = []
        self.num_of_graphs = 0
        self.graphs = {}
        return

    @staticmethod
    def _add_field(results, field, results_dict):
        if field in results.keys():
            results_dict[field] = results[field]

    def add_file(self, filename):
        print("Loading file " + filename)
        try:
            with open(filename, 'rb') as res_file:
                results = pickle.load(res_file)
                self.results.append({})
                fields = ["per_epoch_stats", "layers_names"]
                for field in fields:
                    self._add_field(results=results, field=field, results_dict=self.results[-1])
                self._get_label(filename, results_dict=self.results[-1])
        except FileNotFoundError:
            print("No such file " + filename)
            quit()

    def print_columns(self):
        for result in self.results:
            for col in result["per_epoch_stats"].columns:
                print(col)

    @staticmethod
    def _get_label(filename, results_dict):
        # Load label
        leg_filename = filename
        if leg_filename[-4:] == ".pkl":
            leg_filename = leg_filename[:-4]
        if leg_filename[-6:] == ".stats":
            leg_filename = leg_filename[:-6]
        leg_filename += ".desc.txt"
        try:
            label_file = open(leg_filename, "r")
            results_dict["label"] = label_file.readline().rstrip()
            label_file.close()
        except FileNotFoundError:
            print("Couldnt find " + str(leg_filename) + ", attaching label: " + filename)
            results_dict["label"] = filename

    def _init_graph(self, graph_name=None, title=None, xlabel="", ylabel=""):
        if graph_name in self.graphs.keys():
            return
        self.graphs[graph_name] = {}
        self.graphs[graph_name]["fig"] = fig = plt.figure(figsize=(10, 8))
        self.graphs[graph_name]["main_ax"] = ax = fig.add_subplot(111)
        self.graphs[graph_name]["title"] = title if title else ""
        self.graphs[graph_name]["axes_of_labels"] = {}
        # ax.set_title(args.title + title)
        ax.set_xlabel(xlabel, size=20)
        ax.set_ylabel(ylabel, size=20)

    def _draw_data(self, graph_name=None, xdata=None, ydata=None, label=None, color=None):
        ax = self.graphs[graph_name]["main_ax"]
        self.graphs[graph_name]["axes_of_labels"][label], = ax.plot(xdata, ydata, label=label, color=color, linewidth=3, alpha = 0.8)

    def show_per_epoch_graph(self, graph_name=None, title=None,
                             xfield=None, xdata_override=None, xlabel="",
                             yfield=None, ylabel="", ybottom_max=None, ytop_min=None, save_to_fig=None):
        self._init_graph(graph_name=graph_name, title=title, xlabel=xlabel, ylabel=ylabel)
        ax = self.graphs[graph_name]["main_ax"]
        colors = ['r', 'c', 'b', 'g']
        for i, result in enumerate(self.results):
            label = result["label"]
            xdata = xdata_override
            if xdata is None:
                if xfield not in result["per_epoch_stats"].columns:
                    continue
                xdata = result["per_epoch_stats"][xfield]
            if yfield not in result["per_epoch_stats"].columns:
                continue
            ydata = result["per_epoch_stats"][yfield]
            self._draw_data(graph_name=graph_name, xdata=xdata, ydata=ydata, label=label, color=colors[i])

        if self.graphs[graph_name]["axes_of_labels"].__len__() > 0:
            autoscale_axis_aux(ax, ybottom_max=ybottom_max, ytop_min=ytop_min)
            ax.legend(loc=0)

    def show_graph_with_epoch_slider(self, graph_name=None, title=None,
                                     xfield=None, xdata=None, xlabel="",
                                     yfield=None, ylabel="",
                                     init_epoch=1, ybottom_max=None, ytop_min=None, normalize_sumy=None):
        graph_title = title + " of epoch " + str(init_epoch)
        self._init_graph(graph_name=graph_name, title=graph_title, xlabel=xlabel, ylabel=ylabel)
        self.graphs[graph_name]["fig"].subplots_adjust(bottom=0.25)
        ax = self.graphs[graph_name]["main_ax"]
        max_epoch = 1
        for result in self.results:
            stats = result["per_epoch_stats"]
            max_epoch = int(np.max([max_epoch, stats["epoch_epoch"].max()]))
            fields = []
            if xdata is None:
                fields.append(xfield)
            fields.append(yfield)
            if yfield not in stats.columns:
                continue
            data = stats.loc[lambda df: df.epoch_epoch == init_epoch, fields]
            if data.size < 1:
                continue

            label = result["label"]
            if xdata is None:
                xdata = data[xfield].iloc[0]
            ydata = data[yfield].iloc[0]
            if normalize_sumy is not None:
                ydata = normalize_sumy(ydata)
            self._draw_data(graph_name=graph_name, xdata=xdata, ydata=ydata, label=label)
            self.graphs[graph_name]["axes_of_labels"][label + "maxmarker"] = ax.scatter(
                xdata[0], ydata[0], s=10, c=self.graphs[graph_name]["axes_of_labels"][label].get_color())
        autoscale_axis_aux(ax, ybottom_max=ybottom_max, ytop_min=ytop_min)
        ax.legend(loc=0)

        # Slider
        slider_controller = GraphEpochSliderController(axes_of_labels=self.graphs[graph_name]["axes_of_labels"],
                                                       main_ax=self.graphs[graph_name]["main_ax"],
                                                       fig=self.graphs[graph_name]["fig"],
                                                       results=self.results, yfield=yfield,
                                                       normalize_sumy=normalize_sumy, title=title)
        self.graphs[graph_name]["epoch_slider"] = wdgt.Slider(plt.axes([0.15, 0.1, 0.7, 0.03]),
                                                              label="Epoch", valinit=1, valmin=1,
                                                              valmax=int(max_epoch), valfmt='%d')
        self.graphs[graph_name]["epoch_slider"].on_changed(slider_controller.update)

    def _layer_name(self, layer_idx):
        return self.results[0]["layers_names"][int(layer_idx)]

    def show_graph_with_layer_slider(self, graph_name=None, title=None,
                                     xfield=None, xlabel="",
                                     yfield_prefix="", yfield_suffix="", ylabel="",
                                     init_layer=0, ybottom_max=None, ytop_min=None, layers_names=None):
        if layers_names is None:
            layers_names = self.results[0]["layers_names"]
        graph_title = title + " of layer " + str(init_layer) + " (" + layers_names[init_layer] + ")"
        yfield = yfield_prefix + str(init_layer) + yfield_suffix
        max_layer = 0
        for result in self.results:
            max_layer = [l for l in result["per_epoch_stats"].columns if
                         (l.startswith(yfield_prefix) and l.endswith(yfield_suffix))].__len__() - 1
        self.show_per_epoch_graph(graph_name=graph_name, title=graph_title, xfield=xfield, xlabel=xlabel,
                                  yfield=yfield, ylabel=ylabel, ybottom_max=ybottom_max, ytop_min=ytop_min)
        self.graphs[graph_name]["fig"].subplots_adjust(bottom=0.25)

        # Slider
        slider_controller = GraphLayerSliderController(axes_of_labels=self.graphs[graph_name]["axes_of_labels"],
                                                       main_ax=self.graphs[graph_name]["main_ax"],
                                                       fig=self.graphs[graph_name]["fig"],
                                                       results=self.results,
                                                       xfield=xfield,
                                                       yfield_prefix=yfield_prefix, yfield_suffix=yfield_suffix,
                                                       title=title, layers_names=layers_names)
        self.graphs[graph_name]["layer_slider"] = wdgt.Slider(plt.axes([0.15, 0.1, 0.7, 0.03]),
                                                              label="Layer", valinit=int(init_layer), valmin=0,
                                                              valmax=int(max_layer), valfmt='%d')
        self.graphs[graph_name]["layer_slider"].on_changed(slider_controller.update)

    @staticmethod
    def show():
        plt.show()


class GraphEpochSliderController:
    def __init__(self, axes_of_labels=None, main_ax=None, fig=None, results=None, xfield=None, yfield=None,
                 normalize_sumy=None, title=None):
        self.axes_of_labels = axes_of_labels
        self.results = results
        self.xfield = xfield
        self.yfield = yfield
        self.normalize_sumy = normalize_sumy
        self.main_ax = main_ax
        self.fig = fig
        self.title = title

    def update(self, param):
        epoch = int(param)
        title = self.title + " of epoch " + str(epoch)
        for result in self.results:
            stats = result["per_epoch_stats"]
            label = result["label"]
            fields = []
            if self.yfield:
                fields.append(self.yfield)
            if self.xfield:
                fields.append(self.xfield)
            data = stats.loc[lambda df: df.epoch_epoch == epoch, fields]
            if data.size < 1:
                continue
            if self.xfield:
                xdata = data[self.xfield].iloc[0]
                self.axes_of_labels[label].set_xdata(xdata)
            ydata = data[self.yfield].iloc[0]
            if self.normalize_sumy is not None:
                ydata = self.normalize_sumy(ydata)
            self.axes_of_labels[label].set_ydata(ydata)
            self.axes_of_labels[label + "maxmarker"].remove()
            self.axes_of_labels[label + "maxmarker"] = self.main_ax.scatter(0, ydata[0], s=10,
                                                                            c=self.axes_of_labels[label].get_color())

        self.main_ax.relim()
        autoscale_axis_aux(self.main_ax, ybottom_max=0, xmargin=0.01)
        self.main_ax.set_title(title)
        self.fig.canvas.draw_idle()


class GraphLayerSliderController:
    def __init__(self, axes_of_labels=None, main_ax=None, fig=None, results=None,
                 xfield=None,
                 yfield_prefix=None, yfield_suffix=None,
                 title=None, layers_names=None):
        self.axes_of_labels = axes_of_labels
        self.results = results
        self.xfield = xfield
        self.yfield_prefix = yfield_prefix
        self.yfield_suffix = yfield_suffix
        self.main_ax = main_ax
        self.fig = fig
        self.title = title
        self.layers_names = layers_names

    def update(self, param):
        layer = int(param)
        title = self.title + " of layer " + str(layer) + " (" + self.layers_names[layer] + ")"
        xfield = self.xfield
        yfield = self.yfield_prefix + str(layer) + self.yfield_suffix

        for result in self.results:
            if yfield not in result["per_epoch_stats"].columns:
                continue
            xdata = result["per_epoch_stats"][xfield]
            ydata = result["per_epoch_stats"][yfield]
            label = result["label"]
            self.axes_of_labels[label].set_ydata(ydata)
            self.axes_of_labels[label].set_xdata(xdata)
        self.main_ax.relim()
        autoscale_axis_aux(self.main_ax, ybottom_max=0, xmargin=0)
        self.main_ax.set_title(title)
        self.fig.canvas.draw_idle()


def normalize_to_percent_of_sum(data):
    return (data / np.sum(data)) * 100


font = {'weight': 'bold',
        'size': 16}
font = {'size': 16}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 2
matplotlib.rcParams['xtick.major.width'] = 2
matplotlib.rcParams['xtick.minor.width'] = 2
matplotlib.rcParams['ytick.major.width'] = 2
matplotlib.rcParams['ytick.minor.width'] = 2
matplotlib.rcParams['xtick.major.size'] = 6
matplotlib.rcParams['xtick.minor.size'] = 6
matplotlib.rcParams['ytick.major.size'] = 6
matplotlib.rcParams['ytick.minor.size'] = 6

parser = argparse.ArgumentParser(description='Showing NN training and testing results')
parser.add_argument('--results_files', type=str, nargs='+',
                    help='One or more results file.' +
                         'The description in the legend will be taken from FILENAME.desc.txt (after removing the .pkl)')
parser.add_argument('--title',  type=str, nargs='*', help='Graph title')
parser.add_argument('--print_stats_columns', default=False, action='store_true',
                    help='Use to print per_epoch_stats columns')

args = parser.parse_args()
args.title = ' '.join(args.title) + "\n"

results_pltr = ResultsPlotter()

for filename_ in args.results_files:
    results_pltr.add_file(filename_)

if args.print_stats_columns:
    results_pltr.print_columns()


results_pltr.show_per_epoch_graph(graph_name="acc_per_epc", title="Accuracy per epoch for test set",
                                  xfield="epoch_epoch", xlabel="Epoch",
                                  yfield="test_acc_loss_test_acc", ylabel="Accuracy",
                                  ybottom_max=0, ytop_min=100, save_to_fig="acc.pdf")

results_pltr.show_per_epoch_graph(graph_name="loss_per_epc", title="Loss per epoch for train set",
                                  xfield="epoch_epoch", xlabel="Epoch",
                                  yfield="train_acc_loss_train_loss", ylabel="Loss")

results_pltr.show_per_epoch_graph(graph_name="weights_norm", title="Weights norm",
                                  xfield="epoch_epoch", xlabel="Epoch",
                                  yfield="weight_norm_w_norm", ylabel="Norm",
                                  ybottom_max=0, save_to_fig="weights_norm.pdf")
results_pltr.show_graph_with_layer_slider(graph_name="weights_norm_of_layer",
                                          title="Weights norm of layer",
                                          xfield="epoch_epoch",
                                          xlabel="Epoch",
                                          yfield_prefix="weight_norm_w_norm_layer",
                                          ylabel="Norm",
                                          ybottom_max=0)


results_pltr.show()
