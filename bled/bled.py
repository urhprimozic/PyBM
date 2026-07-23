from pybm.model import Model, Var, Const

model = Model()

phyto_conc = Var("phyto.conc", type="endo", initial=None)
daph_conc = Var("daph.conc", type="exo")
ortp_conc = Var("ortp.conc", type="exo")
silica_conc = Var("silica.conc", type="exo")
no_conc = Var("no.conc", type="exo")
temperature = Var("env.temperature", type="exo")
light = Var("env.light", type="exo")

phyto_max_growth_rate = Const("phyto.maxGrowthRate", initial_value=None)
ortp_half_sat = Const("ortp.halfSaturation", initial_value=None)
silica_half_sat = Const("silica.halfSaturation", initial_value=None)
no_half_sat = Const("no.halfSaturation", initial_value=None)

opt_light = Const("lightInfluence19.optLight", initial_value=None)
growth_ref_temp = Const("tempGrowthLim906.refTemp", initial_value=None)
resp_rate = Const("respirationPP381.respRate", initial_value=None)
resp_ref_temp = Const("tempRespInfluence396.refTemp", initial_value=None)
daph_filtration = Const("daph.maxFiltrationRate", initial_value=None)

model.add(phyto_conc)
model.add(daph_conc)
model.add(ortp_conc)
model.add(silica_conc)
model.add(no_conc)
model.add(temperature)
model.add(light)

model.add(phyto_max_growth_rate)
model.add(ortp_half_sat)
model.add(silica_half_sat)
model.add(no_half_sat)
model.add(opt_light)
model.add(growth_ref_temp)
model.add(resp_rate)
model.add(resp_ref_temp)
model.add(daph_filtration)

def phyto_ode(t, ctx):
    phyto = phyto_conc(t, ctx)
    temp = temperature(t, ctx)
    light_value = light(t, ctx)

    light_lim = (
        light_value
        * np.exp(-light_value / opt_light(ctx) + 1)
        / opt_light(ctx)
    )

    nutrient_lim = (
        ortp_conc(t, ctx) / (ortp_conc(t, ctx) + ortp_half_sat(ctx))
        * silica_conc(t, ctx) / (silica_conc(t, ctx) + silica_half_sat(ctx))
        * no_conc(t, ctx) / (no_conc(t, ctx) + no_half_sat(ctx))
    )

    growth_rate = (
        phyto_max_growth_rate(ctx)
        * temp / growth_ref_temp(ctx)
        * light_lim
        * nutrient_lim
    )

    growth = growth_rate * phyto
    respiration = -resp_rate(ctx) * phyto * temp / resp_ref_temp(ctx)
    grazing = -daph_filtration(ctx) * daph_conc(t, ctx) * phyto

    return growth + respiration + grazing


phyto_conc.ode = phyto_ode