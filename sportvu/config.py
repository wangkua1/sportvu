from _config_section import ConfigSection

import os
REAL_PATH = os.path.dirname(os.path.realpath(__file__))

data = ConfigSection("data")
data.dir = "%s/%s" % (REAL_PATH, "data")

data.config = ConfigSection("data config")
data.config.dir = "%s/%s" % (data.dir, "config")

model = ConfigSection("model")
model.dir = "%s/%s" % (REAL_PATH, "model")

model.config = ConfigSection("model config")
model.config.dir = "%s/%s" % (model.dir, "config")

detect = ConfigSection("detect")
detect.dir = "%s/%s" % (REAL_PATH, "detect")

detect.config = ConfigSection("detect config")
detect.config.dir = "%s/%s" % (detect.dir, "config")

saves = ConfigSection("saves")
saves.dir = "%s/%s" % (REAL_PATH, "saves")

plots = ConfigSection("plots")
plots.dir = "%s/%s" % (REAL_PATH, "plots")

logs = ConfigSection("logs")
logs.dir = "%s/%s" % (REAL_PATH, "logs")
