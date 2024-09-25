from nicegui import events, ui 
import fast
import numpy

CONFIG = {}
CATEGORIES = list(dict.fromkeys([i._category for i in fast.conf.PARAMS]).keys())
ELEMENTS = {}
SIM = None


def load_config(e: events.UploadEventArguments):
    content = e.content.read().decode()
    exec(content, locals())
    c = fast.conf.ConfigParser(locals()['p']).config
    
    for param in c:
        CONFIG[param].value = c[param]
        change_value(CONFIG[param], c[param], ELEMENTS[param])


def add_number_box_bounds(param):
    v = None
    step = None
    if isinstance(param.value, (float, int)):
        v = param.value
        step = param.value/10
    return ui.number(param.name, value=v, suffix=param.unit, min=param._bounds[0], max=param._bounds[1], step=step)


def add_array_box(param):
    boxes = []
    for i, v in enumerate(param.value):
        boxes.append(ui.number(f"{param.name}{i}", value=v, suffix=param.unit, min=param._bounds[0], max=param._bounds[1], step=max(param.value)/10))
    return boxes


def add_param(param):
    elements = []

    with ui.row():

        # some params need special treatment, do that here 
        if param.name in ["NPXLS", "DX"]:
            box = add_number_box_bounds(param)
            checkbox = ui.checkbox("auto", value=(param.value=="auto"))
            return [box, checkbox]
        
        if param.name in ["W0"]:
            box = add_number_box_bounds(param)
            checkbox = ui.checkbox("opt", value=(param.value=="opt"))
            return [box, checkbox]
        
        if param.name in ["LOGFILE"]:
            box = ui.input(param.name)
            checkbox = ui.checkbox("None", value=(param.value==None))
            return [box, checkbox]
        
        if param.name in ["SEED"]:
            box = add_number_box_bounds(param)
            checkbox = ui.checkbox("None", value=(param.value==None))
            return [box, checkbox]
    

        # check through different types of params
        if isinstance(param.value, bool):
            sw = ui.switch(param.name, value=param.value)
            elements.append(sw)
        
        elif isinstance(param.value, str):
            ui.label(param.name)
            rad = ui.radio(param._allowed_values, value=param.value).props("inline")
            elements.append(rad)

        elif isinstance(param.value, (float,int)):
            if param.value < numpy.inf:
                box = add_number_box_bounds(param)
                elements.append(box)
            else:
                box = ui.number(param.name, suffix=param.unit, min=param._bounds[0], max=param._bounds[1])
                cbox = ui.checkbox("inf", value=True)
                elements.append(box)
                elements.append(cbox)
                
            if len(param._allowed_values) > 0:
                for allowed in param._allowed_values:
                    box1 = ui.checkbox(str(allowed), value=False)
                    elements.append(box1)

        elif isinstance(param.value, (numpy.ndarray, list)):
            boxes = add_array_box(param)
            elements = boxes

        elif param.value is None:
            if param._bounds != [-numpy.inf, numpy.inf]:
                # probably a number 
                box = ui.number(param.name, min=param._bounds[0], max=param._bounds[1])
                cbox = ui.checkbox("None", value=True)
                elements.append(box)
                elements.append(cbox)
            else:
                if len(param._allowed_values) > 0:
                    ui.label(param.name)
                    rad = ui.radio(param._allowed_values, value=param.value).props("inline")
                    elements.append(rad)


        else:
            # anything left over? 
            raise Exception(f"Param {param.name} not included in gui init")
            
        return elements
    

def change_value(param, value, elems):
    
    if value in ["auto" or "opt"] or numpy.isinf(value) or value is None:
        elems[0].set_value(None)
        elems[1].set_value(True)
    
    elif value in [True, False] or isinstance(value, str):
        elems[0].set_value(value)

    elif isinstance(value, numpy.ndarray):
        for elem, val in zip(elems, value):
            elem.set_value(val)

    


def handle_change(e, param, elems):
    print(e, param)
    s = e.sender

    value = e.value

    # convert to int for certain values
    if param.name in ["NPXLS", "NITER", "NCHUNKS", "FFTW_THREADS", "SEED"]:
        if type(value) is float:
            value = int(value)

    # different rules for checkboxes
    if type(s) == ui.checkbox:
        if e.value:
            # if checked, set number value to None
            value = s.text
            if value == "None":
                value = None
            elems[0].set_value(None)
        else:
            # if unchecked, force number value to be entered (is this possible?)
            elems[0].run_method("focus")

    try:
        param.value = value
        s.clear()
    except ValueError as err:
        with s:
            ui.tooltip(err.args[0]).classes("bg-red")

    print(param)

def init_sim():
    try:
        SIM = fast.Fast(fast.conf.FastConfig(CONFIG))
    except Exception as e:
        with ui.dialog() as dialog, ui.card():
            ui.label("Error in initialising simulation:")
            ui.label(e.args[0])
            ui.button('Close', on_click=dialog.close)
        dialog.open()


ui.label(f"FAST {fast.__version__}").classes("font-mono text-4xl")

with ui.tabs().classes("w-full") as tabs:
    config_tab = ui.tab("Configuration")
    sim_tab = ui.tab("Simulation")
    results_tab = ui.tab("Results")

with ui.tab_panels(tabs, value=config_tab).classes("w-full"):

    with ui.tab_panel(config_tab):

        ui.button("PRINT DEBUG", on_click=lambda x: print(CONFIG))

        with ui.column().classes("justify-center"):
            ui.upload(label="Load Config File", on_upload=load_config).props('accept=.py')
            
            with ui.row():
                
                for c in CATEGORIES:

                    with ui.card():

                        ui.label(c)

                        for param in fast.conf.PARAMS:

                            CONFIG[param.name] = param

                            if param._category == c:

                                elems = add_param(param)
                                ELEMENTS[param.name] = elems
                                for elem in elems:
                                    elem.on_value_change(lambda e, param=param, 
                                                         elems=elems: handle_change(e, param, elems))
                                    

    with ui.tab_panel(sim_tab):
        ui.button("Initialise", on_click=init_sim)

    with ui.tab_panel(results_tab):
        ui.label("Results")

ui.run()