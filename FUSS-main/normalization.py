import os
import open3d as o3d


def process_single_mesh(input_path, output_path, target_vertices=5000, target_faces=10000):
    """
    处理单个网格文件，使其顶点数为 target_vertices，面数为 target_faces
    :param input_path: 输入网格文件的路径
    :param output_path: 输出网格文件的路径
    :param target_vertices: 目标顶点数，默认为 5000
    :param target_faces: 目标面数，默认为 10000
    """
    try:
        # 读取网格文件
        mesh = o3d.io.read_triangle_mesh(input_path)

        # 简化网格以减少面数
        while len(mesh.triangles) > target_faces:
            mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)

        # 细分网格以增加顶点数
        while len(mesh.vertices) < target_vertices:
            mesh = mesh.subdivide_midpoint(number_of_iterations=1)

        # 再次简化以确保面数符合要求
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)

        # 保存处理后的网格
        o3d.io.write_triangle_mesh(output_path, mesh)
        print(f"已处理 {input_path}，保存到 {output_path}")
    except Exception as e:
        print(f"处理 {input_path} 时出错: {e}")


def batch_process_meshes(input_folder, output_folder, target_vertices=5000, target_faces=10000):
    """
    批量处理文件夹中的网格文件
    :param input_folder: 输入文件夹路径
    :param output_folder: 输出文件夹路径
    :param target_vertices: 目标顶点数，默认为 5000
    :param target_faces: 目标面数，默认为 10000
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith(('.obj', '.off', '.stl')):
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, filename)
            process_single_mesh(input_file, output_file, target_vertices, target_faces)


if __name__ == "__main__":
    input_folder = '/data/vertebrae/off'
    output_folder = '/data/vertebrae/normal'
    batch_process_meshes(input_folder, output_folder)