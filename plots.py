import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from random import randrange

PLOT_FOLDER = 'plots/'

def get_filename(phrase):
  tokens = phrase         \
    .lower()              \
    .replace(",", "")     \
    .replace(".", "")     \
    .split(" ")
  return '%s-#%s.png' % ("-".join(tokens), randrange(100))


def plot_learning_curve(name, xdata, ydata_dict, xname=None, yname=None, baseline=None):
  """ Gets one xdata and a dict of named ydata's to plot
  """
  plt.title(name, fontsize=16)
  plt.xlabel(xname, fontsize=14)
  plt.ylabel(yname, fontsize=14)
  
  for y_data_name, y_data in ydata_dict.iteritems():
    plt.plot(xdata, y_data, label=y_data_name)
  
  plt.legend(loc=4)

  if baseline:
    plt.axhline(y=baseline, color='grey')

  filename = get_filename(name)
  plt.savefig('plots/%s' % filename)
  plt.close()

  # Saving given data just in case
  pickle_filename = filename[:-4]
  xdata.dump("plots/%s-xdata.pkl" % pickle_filename)
  for y_data_name, y_data in ydata_dict.iteritems():
    y_data.dump("plots/%s-ydata-%s.pkl" % ( pickle_filename, y_data_name ) )


def plot_lines(name, xdata, ydata, xname=None, yname=None, axis=None, baseline=None):
  plt.title(name, fontsize=16)
  if xname: 
    plt.xlabel(xname, fontsize=14)
  if yname:
    plt.ylabel(yname, fontsize=14)

  plt.plot(xdata, ydata)
  
  if axis:
    plt.axis(axis)

  if baseline:
    plt.axhline(y=baseline, color='grey')

  filename = get_filename(name)
  plt.savefig('plots/%s' % filename)
  plt.close()
