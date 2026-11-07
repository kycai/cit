DATASET = {
    'flag_y_from_ori': False,
    'flag_X_from_ori': False,
    'flag_X_s_from_ori': False,
    'flag_X_t_from_ori': True, #! based on trial-and-error, True is better
    't_gen_extend_ratio': 0.33, # for generating t_gen by extending the range of t in the original dataset
    
    # Hyperparameter optimization
    'fold_id_out': ['o0', 'o1', 'o2', 'o3', 'o4'], 
    'fold_id_in': ['i0', 'i1', 'i2', 'i3'], # if len(fold_id_out) > 1 then do hyperparameter optimization
}


DICT_PLOT_LABEL_CONV = {
    #########################################
    # Mixture proportion
    'Composite type': r'$\mathrm{CompType}$',
    
    'Water-Binder ratio': r'$Q^{\mathrm{w}}_{\mathrm{m}}/Q^{\mathrm{b}}_{\mathrm{m}}$', # r'$\frac{Q^{\mathrm{w}}_{\mathrm{m}}}{Q^{\mathrm{b}}_{\mathrm{m}}}$', 
    'Binder 1-content': r'$Q^{\mathrm{PC}}_{\mathrm{m}}/Q^{\mathrm{b}}_{\mathrm{m}}$ (%)', # r'$1 - \frac{Q^{\mathrm{SCM}}_{\mathrm{m}}}{Q^{\mathrm{b}}_{\mathrm{m}}}$'
    'Binder 2-content': r'$Q^{\mathrm{SCM-1}}_{\mathrm{m}}/Q^{\mathrm{b}}_{\mathrm{m}}$ (%)', # r'$\frac{Q^{\mathrm{SCM-1}}_{\mathrm{m}}}{Q^{\mathrm{b}}_{\mathrm{m}}}$'
    'Binder 3-content': r'$Q^{\mathrm{SCM-2}}_{\mathrm{m}}/Q^{\mathrm{b}}_{\mathrm{m}}$ (%)', # r'$\frac{Q^{\mathrm{SCM-2}}_{\mathrm{m}}}{Q^{\mathrm{b}}_{\mathrm{m}}}$'
    'Binder 4-content': r'$Q^{\mathrm{SCM-3}}_{\mathrm{m}}/Q^{\mathrm{b}}_{\mathrm{m}}$ (%)', # r'$\frac{Q^{\mathrm{SCM-3}}_{\mathrm{m}}}{Q^{\mathrm{b}}_{\mathrm{m}}}$'
    
    'Replacement rate': r'$r_{\mathrm{r}}$ (%)', # r'$\frac{Q^{\mathrm{SCM}}_{\mathrm{m}}}{Q^{\mathrm{b}}_{\mathrm{m}}}$'
    'Binder 1-id': 'PC type',
    'Binder 2-id': r'$\mathrm{SCM-1}$ type',
    'Binder 3-id': r'$\mathrm{SCM-2}$ type',
    'Binder 4-id': r'$\mathrm{SCM-3}$ type',

    'Fine aggregate-content': r'$Q^{\mathrm{FA}}_{\mathrm{m}}/Q^{\mathrm{s}}_{\mathrm{m}}$ (%)', # r'$\frac{Q^{\mathrm{FA}}_{\mathrm{m}}}{Q^{\mathrm{b+FA+CA}}_{\mathrm{m}}}$'
    'Fine aggregate-content unit': r'$Q^{\mathrm{FA}}_{\mathrm{m}}/Q^{\mathrm{s}}_{\mathrm{m}}$ (% $\mathrm{ Unit}$)',
    'Coarse aggregate-content': r'$Q^{\mathrm{CA}}_{\mathrm{m}}/Q^{\mathrm{s}}_{\mathrm{m}}$ (%)', # r'$\frac{Q^{\mathrm{CA}}_{\mathrm{m}}}{Q^{\mathrm{b+FA+CA}}_{\mathrm{m}}}$'
    'Coarse aggregate-content unit': r'$Q^{\mathrm{FA}}_{\mathrm{m}}/Q^{\mathrm{s}}_{\mathrm{m}}$ (% $\mathrm{ Unit}$)',

    'Mixture design-binder 1': r'$Q^{\mathrm{PC}}_{\mathrm{m}}$ ($\mathrm{kg/m^3}$ c.)',
    'Mixture design-binder 2': r'$Q^{\mathrm{SCM-1}}_{\mathrm{m}} ($\mathrm{kg/m^3}$ c.)$',
    'Mixture design-binder 3': r'$Q^{\mathrm{SCM-2}}_{\mathrm{m}} ($\mathrm{kg/m^3}$ c.)$',
    'Mixture design-binder 4': r'$Q^{\mathrm{SCM-3}}_{\mathrm{m}} ($\mathrm{kg/m^3}$ c.)$',
    'Mixture design-water': r'$Q^{\mathrm{w}}_{\mathrm{m}}$ ($\mathrm{kg/m^3}$ c.)',
    'Mixture design-fine aggregate': r'$Q^{\mathrm{FA}}_{\mathrm{m}}$ ($\mathrm{kg/m^3}$ c.)',
    'Mixture design-coarse aggregate': r'$Q^{\mathrm{CA}}_{\mathrm{m}}$ ($\mathrm{kg/m^3}$ c.)',
    'Mixture design-targeted density': r'$\rho^{\mathrm{c}}$ ($\mathrm{kg/m^3}$)',
    'Mixture design-targeted air content': r'$\mathrm{air}$ (vol% c.)',

    'Binder 1_Mixture design-binder': r'$Q^{\mathrm{PC}}_{\mathrm{m}}$ ($\mathrm{kg/m^3}$ c.)',
    'Binder 234_Mixture design-binder': r'$Q^{\mathrm{SCM}}_{\mathrm{m}}$ ($\mathrm{kg/m^3}$ c.)',

    # Specimen geometry
    'Scheme-specimening': r'$\mathrm{Scheme }_{\mathrm{Spec}}$',
    'Specimen geometric type': r'Specimen shape',
    'Specimen length': r'$L$ (mm)', 
    'Specimen width': r'$W$ (mm)', 
    'Specimen height': r'$H$ (mm)', 
    'Specimen compaction pressure': r'$\mathrm{CompactPress}$',
    
    # Curing condition
    'Scheme-curing': r'$\mathrm{Scheme }_{\mathrm{NC}}$',
    'Curing temperature': r'$T$ ($^\circ\mathrm{C}$)', # r'$T_{\mathrm{NC}}$',
    'Curing relative humidity': r'$RH_{\mathrm{NC}}$ (%)',
    'Curing duration': r'$t$ (d)', #!!! r'$t_{\mathrm{NC}}$'
    
    #########################################
    # Mechanical properties
    'Compression loading control type': r'$\mathrm{LoadControl }_{\mathrm{Comp}}$',
    'Compression loading rate': r'$v_{\mathrm{Load}}$', 
    'Compressive strength': r'$f_{\mathrm{c}}$ (MPa)', 
    
    #########################################
    # Chemical admixture
    'Chemical admixture 1_Classification': r'$\mathrm{AdmixType }_{\mathrm{CA1}}$',
    'Chemical admixture 2_Classification': r'$\mathrm{AdmixType }_{\mathrm{CA2}}$',
    'Chemical admixture 3_Classification': r'$\mathrm{AdmixType }_{\mathrm{CA3}}$',
    
    #########################################
    # Aggregates
    'Fine aggregate_Specific gravity': r'$\rho^{\mathrm{FA}}/\rho^{\mathrm{w}}$ ',
    'Coarse aggregate_Specific gravity': r'$\rho^{\mathrm{CA}}/\rho^{\mathrm{w}}$',

    #########################################
    # Binder: PC
    'Binder 1_Dv,10': r'$D_{\mathrm{v, 10}}^{\mathrm{PC}}$ ($\mathrm{\mu m}$)', 
    'Binder 1_Dv,50': r'$D_{\mathrm{v, 50}}^{\mathrm{PC}}$ ($\mathrm{\mu m}$)', 
    'Binder 1_Dv,90': r'$D_{\mathrm{v, 90}}^{\mathrm{PC}}$ ($\mathrm{\mu m}$)', 
    
    'Binder 1_Specific surface area': r'$SSA^{\mathrm{PC}}$ ($\mathrm{m^2/kg}$)', 
    
    'Binder 1_Skeletal density': r'$\rho^{\mathrm{PC}}$ ($\mathrm{g/cm^3}$)', 
    
    'Binder 1_Na2O': r'$C_{\mathrm{m, Na_2O}}^{\mathrm{PC}}$ (%)', 
    'Binder 1_MgO': r'$C_{\mathrm{m, MgO}}^{\mathrm{PC}}$ (%)', 
    'Binder 1_Al2O3': r'$C_{\mathrm{m, Al_2O_3}}^{\mathrm{PC}}$ (%)', 
    'Binder 1_SiO2': r'$C_{\mathrm{m, SiO_2}}^{\mathrm{PC}}$ (%)', 
    'Binder 1_SO3': r'$C_{\mathrm{m, SO_3}}^{\mathrm{PC}}$ (%)', 
    'Binder 1_K2O': r'$C_{\mathrm{m, K_2O}}^{\mathrm{PC}}$ (%)', 
    'Binder 1_CaO': r'$C_{\mathrm{m, CaO}}^{\mathrm{PC}}$ (%)', 
    'Binder 1_Fe2O3': r'$C_{\mathrm{m, Fe_2O_3}}^{\mathrm{PC}}$ (%)', 
    
    'Binder 1_Loss on ignition': r'$LOI^{\mathrm{PC}}$ (%)', 
    
    'Binder 1_Water-Soluble alkalies': r'$\mathrm{Na_{2}O_{eq}}^{\mathrm{PC}}$ (%)', 
    
    'Binder 1_Amorphous+others': r'$P_{\mathrm{m, Amorph}}^{\mathrm{PC}}$ (%)', 
    'Binder 1_Crystalline': r'$P_{\mathrm{m, Crystal}}^{\mathrm{PC}}$ (%)', 
    
    'Binder 1_Crystalline': r'$P_{\mathrm{m, Crystal}}^{\mathrm{PC}}$ (%)',
    'Binder 1_Åkermanite': r'$P_{\mathrm{m, Åkermanite}}^{\mathrm{PC}}$ (%)',
    'Binder 1_Albite': r'$P_{\mathrm{m, Albite}}^{\mathrm{PC}}$ (%)',
    'Binder 1_Anhydrite': r'$P_{\mathrm{m, Anhydrite}}^{\mathrm{PC}}$ (%)',
    'Binder 1_Ankerite': r'$P_{\mathrm{m, Ankerite}}^{\mathrm{PC}}$ (%)',
    'Binder 1_Anorthite': r'$P_{\mathrm{m, Anorthite}}^{\mathrm{PC}}$ (%)',
    'Binder 1_Aphthitalite': r'$P_{\mathrm{m, Aphthitalite}}^{\mathrm{PC}}$ (%)',
    'Binder 1_Arcanite': r'$P_{\mathrm{m, Arcanite}}^{\mathrm{PC}}$ (%)',
    'Binder 1_Augite': r'$P_{\mathrm{m, Augite}}^{\mathrm{PC}}$ (%)',
    'Binder 1_Bassanite': r'$P_{\mathrm{m, Bassanite}}^{\mathrm{PC}}$ (%)',
    'Binder 1_Burkeite': r'$P_{\mathrm{m, Burkeite}}^{\mathrm{PC}}$ (%)',
    'Binder 1_C2S': r'$P_{\mathrm{m, C2S}}^{\mathrm{PC}}$ (%)',
    'Binder 1_C2S-alpha': r'$P_{\mathrm{m, C2S-\alpha}}^{\mathrm{PC}}$ (%)',
    "Binder 1_C2S-alpha'": r'$P_{\mathrm{m, C2S-\alpha\prime}}^{\mathrm{PC}}$ (%)',
    'Binder 1_C2S-beta': r'$P_{\mathrm{m, C2S-\beta}}^{\mathrm{PC}}$ (%)',
    'Binder 1_C3A': r'$P_{\mathrm{m, C3A}}^{\mathrm{PC}}$ (%)',
    'Binder 1_C3A-cubic': r'$P_{\mathrm{m, C3A-cubic}}^{\mathrm{PC}}$ (%)',
    'Binder 1_C3A-orthorhombic': r'$P_{\mathrm{m, C3A-orthorhombic}}^{\mathrm{PC}}$ (%)',
    'Binder 1_C3S': r'$P_{\mathrm{m, C3S}}^{\mathrm{PC}}$ (%)',
    'Binder 1_C4AF': r'$P_{\mathrm{m, C4AF}}^{\mathrm{PC}}$ (%)',
    'Binder 1_Calcite': r'$P_{\mathrm{m, Calcite}}^{\mathrm{PC}}$ (%)',
    'Binder 1_Clinoptilolite-Na': r'$P_{\mathrm{m, Clinoptilolite-Na}}^{\mathrm{PC}}$ (%)',
    'Binder 1_Cristobalite': r'$P_{\mathrm{m, Cristobalite}}^{\mathrm{PC}}$ (%)',
    'Binder 1_Diopside': r'$P_{\mathrm{m, Diopside}}^{\mathrm{PC}}$ (%)',
    'Binder 1_Dolomite': r'$P_{\mathrm{m, Dolomite}}^{\mathrm{PC}}$ (%)',
    'Binder 1_Gehlenite': r'$P_{\mathrm{m, Gehlenite}}^{\mathrm{PC}}$ (%)',
    'Binder 1_Gypsum': r'$P_{\mathrm{m, Gypsum}}^{\mathrm{PC}}$ (%)',
    'Binder 1_Halite-K': r'$P_{\mathrm{m, Halite-K}}^{\mathrm{PC}}$ (%)',
    'Binder 1_Hannebachite': r'$P_{\mathrm{m, Hannebachite}}^{\mathrm{PC}}$ (%)',
    'Binder 1_Hematite': r'$P_{\mathrm{m, Hematite}}^{\mathrm{PC}}$ (%)',
    'Binder 1_Kalsilite': r'$P_{\mathrm{m, Kalsilite}}^{\mathrm{PC}}$ (%)',
    'Binder 1_Lime': r'$P_{\mathrm{m, Lime}}^{\mathrm{PC}}$ (%)',
    'Binder 1_Maghemite': r'$P_{\mathrm{m, Maghemite}}^{\mathrm{PC}}$ (%)',
    'Binder 1_Magnesium silicate': r'$P_{\mathrm{m, Magnesium\ silicate}}^{\mathrm{PC}}$ (%)',
    'Binder 1_Magnetite': r'$P_{\mathrm{m, Magnetite}}^{\mathrm{PC}}$ (%)',
    'Binder 1_Mayenite': r'$P_{\mathrm{m, Mayenite}}^{\mathrm{PC}}$ (%)',
    'Binder 1_Merwinite': r'$P_{\mathrm{m, Merwinite}}^{\mathrm{PC}}$ (%)',
    'Binder 1_Mullite': r'$P_{\mathrm{m, Mullite}}^{\mathrm{PC}}$ (%)',
    'Binder 1_Muscovite or Illite': r'$P_{\mathrm{m, Muscovite/Illite}}^{\mathrm{PC}}$ (%)',
    'Binder 1_Periclase': r'$P_{\mathrm{m, Periclase}}^{\mathrm{PC}}$ (%)',
    'Binder 1_Portlandite': r'$P_{\mathrm{m, Portlandite}}^{\mathrm{PC}}$ (%)',
    'Binder 1_Quartz': r'$P_{\mathrm{m, Quartz}}^{\mathrm{PC}}$ (%)',
    'Binder 1_Siderite': r'$P_{\mathrm{m, Siderite}}^{\mathrm{PC}}$ (%)',
    'Binder 1_Sodium sulfate': r'$P_{\mathrm{m, Sodium\ sulfate}}^{\mathrm{PC}}$ (%)',
    'Binder 1_Srebrodolskite': r'$P_{\mathrm{m, Srebrodolskite}}^{\mathrm{PC}}$ (%)',
    'Binder 1_Talc': r'$P_{\mathrm{m, Talc}}^{\mathrm{PC}}$ (%)',

    'Binder 1_Cumulative released heat-7dR3': r'$H_{\mathrm{R3, 7d}}^{\mathrm{PC}}$ (J/g material)', 
    'Binder 1_Consumed Ca(OH)2-7dR3': r'$CH_{\mathrm{R3, 7d}}^{\mathrm{PC}}$ (g/100g material)', 
    'Binder 1_Cumulative released heat-10dmR3': r'$H_{\mathrm{mR3, 10d}}^{\mathrm{PC}}$ (J/g material)', 
    'Binder 1_Consumed Ca(OH)2-10dmR3': r'$CH_{\mathrm{mR3, 10d}}^{\mathrm{PC}}$ (g/100g material)', 

    #########################################
    # Binder: SCM
    'Binder 234_Dv,10': r'$D_{\mathrm{v, 10}}^{\mathrm{SCM}}$ ($\mathrm{\mu m}$)', 
    'Binder 234_Dv,50': r'$D_{\mathrm{v, 50}}^{\mathrm{SCM}}$ ($\mathrm{\mu m}$)', 
    'Binder 234_Dv,90': r'$D_{\mathrm{v, 90}}^{\mathrm{SCM}}$ ($\mathrm{\mu m}$)', 
    
    'Binder 234_Specific surface area': r'$SSA^{\mathrm{SCM}}$ ($\mathrm{m^2/kg}$)', 
    
    'Binder 234_Skeletal density': r'$\rho^{\mathrm{SCM}}$ ($\mathrm{g/cm^3}$)', 
    
    'Binder 234_Na2O': r'$C_{\mathrm{m, Na_2O}}^{\mathrm{SCM}}$ (%)', 
    'Binder 234_MgO': r'$C_{\mathrm{m, MgO}}^{\mathrm{SCM}}$ (%)', 
    'Binder 234_Al2O3': r'$C_{\mathrm{m, Al_2O_3}}^{\mathrm{SCM}}$ (%)', 
    'Binder 234_SiO2': r'$C_{\mathrm{m, SiO_2}}^{\mathrm{SCM}}$ (%)', 
    'Binder 234_SO3': r'$C_{\mathrm{m, SO_3}}^{\mathrm{SCM}}$ (%)', 
    'Binder 234_K2O': r'$C_{\mathrm{m, K_2O}}^{\mathrm{SCM}}$ (%)', 
    'Binder 234_CaO': r'$C_{\mathrm{m, CaO}}^{\mathrm{SCM}}$ (%)', 
    'Binder 234_Fe2O3': r'$C_{\mathrm{m, Fe_2O_3}}^{\mathrm{SCM}}$ (%)', 
    
    'Binder 234_Loss on ignition': r'$LOI^{\mathrm{SCM}}$ (%)', 
    
    'Binder 234_Water-Soluble alkalies': r'$\mathrm{Na_{2}O_{eq}}^{\mathrm{SCM}}$ (%)', 
    
    'Binder 234_Amorphous+others': r'$P_{\mathrm{m, Amorph}}^{\mathrm{SCM}}$ (%)', 
    'Binder 234_Crystalline': r'$P_{\mathrm{m, Crystal}}^{\mathrm{SCM}}$ (%)', 
    
    'Binder 234_Crystalline': r'$P_{\mathrm{m, Crystal}}^{\mathrm{SCM}}$ (%)',
    'Binder 234_Åkermanite': r'$P_{\mathrm{m, Åkermanite}}^{\mathrm{SCM}}$ (%)',
    'Binder 234_Albite': r'$P_{\mathrm{m, Albite}}^{\mathrm{SCM}}$ (%)',
    'Binder 234_Anhydrite': r'$P_{\mathrm{m, Anhydrite}}^{\mathrm{SCM}}$ (%)',
    'Binder 234_Ankerite': r'$P_{\mathrm{m, Ankerite}}^{\mathrm{SCM}}$ (%)',
    'Binder 234_Anorthite': r'$P_{\mathrm{m, Anorthite}}^{\mathrm{SCM}}$ (%)',
    'Binder 234_Aphthitalite': r'$P_{\mathrm{m, Aphthitalite}}^{\mathrm{SCM}}$ (%)',
    'Binder 234_Arcanite': r'$P_{\mathrm{m, Arcanite}}^{\mathrm{SCM}}$ (%)',
    'Binder 234_Augite': r'$P_{\mathrm{m, Augite}}^{\mathrm{SCM}}$ (%)',
    'Binder 234_Bassanite': r'$P_{\mathrm{m, Bassanite}}^{\mathrm{SCM}}$ (%)',
    'Binder 234_Burkeite': r'$P_{\mathrm{m, Burkeite}}^{\mathrm{SCM}}$ (%)',
    'Binder 234_C2S': r'$P_{\mathrm{m, C2S}}^{\mathrm{SCM}}$ (%)',
    'Binder 234_C2S-alpha': r'$P_{\mathrm{m, C2S-\alpha}}^{\mathrm{SCM}}$ (%)',
    "Binder 234_C2S-alpha'": r'$P_{\mathrm{m, C2S-\alpha\prime}}^{\mathrm{SCM}}$ (%)',
    'Binder 234_C2S-beta': r'$P_{\mathrm{m, C2S-\beta}}^{\mathrm{SCM}}$ (%)',
    'Binder 234_C3A': r'$P_{\mathrm{m, C3A}}^{\mathrm{SCM}}$ (%)',
    'Binder 234_C3A-cubic': r'$P_{\mathrm{m, C3A-cubic}}^{\mathrm{SCM}}$ (%)',
    'Binder 234_C3A-orthorhombic': r'$P_{\mathrm{m, C3A-orthorhombic}}^{\mathrm{SCM}}$ (%)',
    'Binder 234_C3S': r'$P_{\mathrm{m, C3S}}^{\mathrm{SCM}}$ (%)',
    'Binder 234_C4AF': r'$P_{\mathrm{m, C4AF}}^{\mathrm{SCM}}$ (%)',
    'Binder 234_Calcite': r'$P_{\mathrm{m, Calcite}}^{\mathrm{SCM}}$ (%)',
    'Binder 234_Clinoptilolite-Na': r'$P_{\mathrm{m, Clinoptilolite-Na}}^{\mathrm{SCM}}$ (%)',
    'Binder 234_Cristobalite': r'$P_{\mathrm{m, Cristobalite}}^{\mathrm{SCM}}$ (%)',
    'Binder 234_Diopside': r'$P_{\mathrm{m, Diopside}}^{\mathrm{SCM}}$ (%)',
    'Binder 234_Dolomite': r'$P_{\mathrm{m, Dolomite}}^{\mathrm{SCM}}$ (%)',
    'Binder 234_Gehlenite': r'$P_{\mathrm{m, Gehlenite}}^{\mathrm{SCM}}$ (%)',
    'Binder 234_Gypsum': r'$P_{\mathrm{m, Gypsum}}^{\mathrm{SCM}}$ (%)',
    'Binder 234_Halite-K': r'$P_{\mathrm{m, Halite-K}}^{\mathrm{SCM}}$ (%)',
    'Binder 234_Hannebachite': r'$P_{\mathrm{m, Hannebachite}}^{\mathrm{SCM}}$ (%)',
    'Binder 234_Hematite': r'$P_{\mathrm{m, Hematite}}^{\mathrm{SCM}}$ (%)',
    'Binder 234_Kalsilite': r'$P_{\mathrm{m, Kalsilite}}^{\mathrm{SCM}}$ (%)',
    'Binder 234_Lime': r'$P_{\mathrm{m, Lime}}^{\mathrm{SCM}}$ (%)',
    'Binder 234_Maghemite': r'$P_{\mathrm{m, Maghemite}}^{\mathrm{SCM}}$ (%)',
    'Binder 234_Magnesium silicate': r'$P_{\mathrm{m, Magnesium\ silicate}}^{\mathrm{SCM}}$ (%)',
    'Binder 234_Magnetite': r'$P_{\mathrm{m, Magnetite}}^{\mathrm{SCM}}$ (%)',
    'Binder 234_Mayenite': r'$P_{\mathrm{m, Mayenite}}^{\mathrm{SCM}}$ (%)',
    'Binder 234_Merwinite': r'$P_{\mathrm{m, Merwinite}}^{\mathrm{SCM}}$ (%)',
    'Binder 234_Mullite': r'$P_{\mathrm{m, Mullite}}^{\mathrm{SCM}}$ (%)',
    'Binder 234_Muscovite or Illite': r'$P_{\mathrm{m, Muscovite/Illite}}^{\mathrm{SCM}}$ (%)',
    'Binder 234_Periclase': r'$P_{\mathrm{m, Periclase}}^{\mathrm{SCM}}$ (%)',
    'Binder 234_Portlandite': r'$P_{\mathrm{m, Portlandite}}^{\mathrm{SCM}}$ (%)',
    'Binder 234_Quartz': r'$P_{\mathrm{m, Quartz}}^{\mathrm{SCM}}$ (%)',
    'Binder 234_Siderite': r'$P_{\mathrm{m, Siderite}}^{\mathrm{SCM}}$ (%)',
    'Binder 234_Sodium sulfate': r'$P_{\mathrm{m, Sodium\ sulfate}}^{\mathrm{SCM}}$ (%)',
    'Binder 234_Srebrodolskite': r'$P_{\mathrm{m, Srebrodolskite}}^{\mathrm{SCM}}$ (%)',
    'Binder 234_Talc': r'$P_{\mathrm{m, Talc}}^{\mathrm{SCM}}$ (%)',
    
    'Binder 234_Cumulative released heat-7dR3': r'$H_{\mathrm{R3, 7d}}^{\mathrm{SCM}}$ (J/g material)', 
    'Binder 234_Consumed Ca(OH)2-7dR3': r'$CH_{\mathrm{R3, 7d}}^{\mathrm{SCM}}$ (g/100g material)', 
    'Binder 234_Cumulative released heat-10dmR3': r'$H_{\mathrm{mR3, 10d}}^{\mathrm{SCM}}$ (J/g material)', 
    'Binder 234_Consumed Ca(OH)2-10dmR3': r'$CH_{\mathrm{mR3, 10d}}^{\mathrm{SCM}}$ (g/100g material)',
    
    }
