import os, sys
import hazelbean as hb
import urban_mental_health_tasks


def build_task_tree(p):
    p.task_reproject_and_resample_lulc_to_population_grid = p.add_task(urban_mental_health_tasks.task_warp_lulc_to_population)
    p.task_convert_lulc_to_ndvi_baseline = p.add_task(urban_mental_health_tasks.task_convert_lulc_to_ndvi_baseline)
    p.task_create_no_vegetation_scenario = p.add_task(urban_mental_health_tasks.task_create_no_vegetation_scenario)
    p.task_calculate_delta_nature_exposure = p.add_task(urban_mental_health_tasks.task_calculate_delta_nature_exposure)
    p.task_calculate_preventable_cases = p.add_task(urban_mental_health_tasks.task_calculate_preventable_cases)
    p.task_rasterize_regions_to_population_grid = p.add_task(urban_mental_health_tasks.task_rasterize_regions_to_population_grid)
    p.task_aggregate_preventable_cases_by_region = p.add_task(urban_mental_health_tasks.task_aggregate_preventable_cases_by_region)
    p.task_calculate_country_costs = p.add_task(urban_mental_health_tasks.task_calculate_country_costs)


if __name__ == '__main__':

    # Create the project flow object
    p = hb.ProjectFlow()


    # Set directories
    p.user_dir = os.path.expanduser('~')
    p.extra_dirs = ['Files', 'global_invest', 'projects']
    p.project_name = 'urban_mental_health_' + hb.pretty_time()
    #p.project_name = 'urban_mental_health_20261105_221636'
    p.project_dir = os.path.join(p.user_dir, os.sep.join(p.extra_dirs), p.project_name)
    p.set_project_dir(p.project_dir)

    # Set base_data_dir. Will download required files here.
    p.base_data_dir = os.path.join(p.user_dir, 'Files', 'base_data', 'submissions', 'urban_mental_health')

    # Set model paths for global processing
    p.aoi = 'global'

    p.base_year_lulc_path = p.get_path(os.path.join(p.base_data_dir, '../../lulc/glc_fcs30d/lulc_glc_fcs30d_2019.tif'))  # 30m LULC
    #p.base_year_lulc_path = p.get_path('/scratch.global/sethi065/Files/base_data/lulc/lulc_glc_fcs30d_2019.tif')  # 30m LULC
    p.population_2019_path = p.get_path(os.path.join(p.base_data_dir, 'worldpop/100m_2019_global-mosaic.tif'))  # 100m population
    #p.population_2019_path = p.get_path('/scratch.global/sethi065/Files/base_data/population/worldpop/100m_2019_global.tif')  # 100m population
    p.lulc_attribute_table_path = p.get_path(os.path.join(p.base_data_dir, 'glc_fcs30d_attribute_table_processed.csv'))
    p.effect_size_table_path = p.get_path(os.path.join(p.base_data_dir, 'effect_size.xlsx'))  # odds ratio for depression
    p.baseline_prevalence_rate = 0.05  # global average prevalence
    p.prevalence_nonexposed = 0.05  # depression risk in the reference (less-green) group
    # prevalence_nonexposed is p_0 used to convert Liu et al. 2023's OR to RR;
    # = baseline prevalence so p0 and baseline cases share one population and outcome definition
    p.boundary_path = p.get_path(os.path.join(p.base_data_dir, 'urban_suburbs_denserural_boundaries_2019.gpkg'))  # GUPPD + GHS-SMOD classes 21 and 13
    p.health_cost_rate_path = p.get_path(os.path.join(p.base_data_dir, 'treatment_cost.csv'))

    # Build the task tree and execute it.
    build_task_tree(p)
    p.execute()

    print("URBAN MENTAL HEALTH MODEL COMPLETED")
