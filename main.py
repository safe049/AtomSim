import pygame
import sys
import random
import math
import json
from pygame.locals import *

# 初始化pygame
pygame.init()

# 屏幕设置
WIDTH, HEIGHT = 1024, 768
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('高级原子拟真器')

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
COLORS = [
    (255, 0, 0),    # 红
    (0, 255, 0),    # 绿
    (0, 0, 255),    # 蓝
    (255, 255, 0),  # 黄
    (255, 0, 255),  # 紫
    (0, 255, 255),  # 青
    (255, 128, 0),  # 橙
    (128, 0, 255)   # 紫罗兰
]

# 发音表
CONSONANTS = ["Zh", "Ch", "Sh", "B", "P", "M", "F", "D", "T", "N", "L", "G", "K", "H", "J", "Q", "X", "R", "Z", "C", "S"]
VOWELS = ["A", "O", "E", "I", "U"]

# 元素类型和属性
ELEMENT_TYPES = {
    "金属": {"color": (200, 200, 200), "radius": 25, "valence": 1, "mass": 30, "electronegativity": 0.7},
    "非金属": {"color": (0, 200, 200), "radius": 20, "valence": 3, "mass": 20, "electronegativity": 3.0},
    "惰性气体": {"color": (200, 0, 200), "radius": 30, "valence": 0, "mass": 40, "electronegativity": 0.0},
    "卤素": {"color": (200, 200, 0), "radius": 22, "valence": 1, "mass": 25, "electronegativity": 2.8}
}
show_atom_names = True  # 初始状态为显示原子名字
show_charge = True  # 初始状态为显示电荷

# 原子类
class Atom:
    def __init__(self, x, y, element_type=None):
        self.x = x
        self.y = y
        self.world_width = WIDTH * 3  # 扩大世界边界
        self.world_height = HEIGHT * 3
        
        # 随机选择元素类型或使用指定的
        if element_type is None:
            self.type = random.choice(list(ELEMENT_TYPES.keys()))
        else:
            self.type = element_type
            
        self.radius = ELEMENT_TYPES[self.type]["radius"] + random.randint(-5, 5)
        self.color = ELEMENT_TYPES[self.type]["color"]
        self.valence_electrons = ELEMENT_TYPES[self.type]["valence"]
        self.mass = ELEMENT_TYPES[self.type]["mass"] + random.randint(-5, 5)
        self.electronegativity = ELEMENT_TYPES[self.type]["electronegativity"] * random.uniform(0.9, 1.1)
        
        self.vx = random.uniform(-2, 2)
        self.vy = random.uniform(-2, 2)
        self.charge = 0  # 初始电荷为0，由电子数决定
        self.electrons = random.randint(0, min(8, self.valence_electrons * 2))
        self.name = self.generate_name()
        self.bonds = []  # 存储与其他原子的化学键
        self.bond_capacity = self.valence_electrons  # 最大键数
        self.bond_length = self.radius * 2.5  # 键长
        self.selected = False  # 是否被选中
        self.id = random.randint(0, 999999)  # 唯一标识符
        self.electron_transfer_timer = 0  # 电子转移计时器
    
    def generate_name(self):
        # 随机生成原子名字
        consonant = random.choice(CONSONANTS)
        vowel = random.choice(VOWELS)
        suffix = random.choice(["", "ium", "on", "gen", "ide"])
        return consonant + vowel + suffix
    
    def update(self, atoms):
        # 更新电子转移计时器
        if self.electron_transfer_timer > 0:
            self.electron_transfer_timer -= 1
        
        # 更新电荷状态（基于实际电子数）
        self.charge = self.valence_electrons - self.electrons
        
        # 应用所有力
        self.apply_forces(atoms)
        
        # 更新位置
        self.x += self.vx
        self.y += self.vy
        
        # 周期性边界条件（模拟无限空间）
        if self.x < 0:
            self.x += self.world_width
        elif self.x > self.world_width:
            self.x -= self.world_width
        if self.y < 0:
            self.y += self.world_height
        elif self.y > self.world_height:
            self.y -= self.world_height
    
    def apply_forces(self, atoms):
        # 重置加速度
        ax, ay = 0, 0
        
        # 计算所有原子间的作用力
        for other in atoms:
            if other != self:
                # 考虑周期性边界条件的最小距离
                dx = other.x - self.x
                dy = other.y - self.y
                
                # 考虑周期性边界
                if dx > self.world_width / 2:
                    dx -= self.world_width
                elif dx < -self.world_width / 2:
                    dx += self.world_width
                if dy > self.world_height / 2:
                    dy -= self.world_height
                elif dy < -self.world_height / 2:
                    dy += self.world_height
                
                distance_sq = dx*dx + dy*dy
                distance = max(math.sqrt(distance_sq), 1)
                
                # 库仑力 (电荷相互作用)
                coulomb_force = (self.charge * other.charge) / (distance_sq + 100) * 0.5
                
                # 范德华力 (近距离排斥，远距离吸引)
                vdw_force = 0
                if distance < self.radius + other.radius + 20:
                    # 近距离排斥
                    vdw_force = -1000 / (distance_sq + 1)
                elif distance < 200:
                    # 中等距离微弱吸引
                    vdw_force = 5 / (distance + 1)
                
                # 化学键力 (如果已形成键)
                bond_force = 0
                if other in [bond[0] for bond in self.bonds]:
                    # 键的弹性力 (类似弹簧)
                    desired_distance = self.bond_length + other.bond_length
                    bond_strength = 0.3
                    bond_force = bond_strength * (distance - desired_distance)
                
                # 总力
                total_force = coulomb_force + vdw_force + bond_force
                
                # 应用力 (F=ma)
                if distance > 0:
                    fx = total_force * dx / distance
                    fy = total_force * dy / distance
                    ax += fx / self.mass
                    ay += fy / self.mass
        
        # 更新速度
        self.vx += ax
        self.vy += ay
        
        # 速度阻尼
        self.vx *= 0.99
        self.vy *= 0.99
    
    def form_bond(self, other_atom):
        # 检查是否可以形成键
        if (len(self.bonds) < self.bond_capacity and 
            len(other_atom.bonds) < other_atom.bond_capacity and
            other_atom not in [bond[0] for bond in self.bonds]):
            
            # 计算键强度 (基于电负性差异)
            electroneg_diff = abs(self.electronegativity - other_atom.electronegativity)
            bond_strength = min(1.0, electroneg_diff / 3.0)  # 电负性差异越大，键越强
            
            # 添加键
            self.bonds.append((other_atom, bond_strength))
            other_atom.bonds.append((self, bond_strength))
            
            # 尝试电子转移 (仅在电负性差异较大时)
            if electroneg_diff > 1.0 and self.electron_transfer_timer == 0 and other_atom.electron_transfer_timer == 0:
                self.transfer_electron(other_atom)
            
            return True
        return False
    
    def transfer_electron(self, other_atom):
        """尝试转移电子到另一个原子"""
        # 确定电子转移方向 (从电负性低的原子到电负性高的原子)
        if self.electronegativity > other_atom.electronegativity:
            donor = other_atom
            acceptor = self
        else:
            donor = self
            acceptor = other_atom
        
        # 检查是否满足转移条件
        if donor.electrons > 0 and acceptor.electrons < acceptor.valence_electrons * 2:  # 允许接受额外电子
            # 转移电子
            donor.electrons -= 1
            acceptor.electrons += 1
            
            # 设置转移冷却时间
            donor.electron_transfer_timer = 60  # 1秒冷却
            acceptor.electron_transfer_timer = 60
            
            # 更新电荷状态
            donor.charge = donor.valence_electrons - donor.electrons
            acceptor.charge = acceptor.valence_electrons - acceptor.electrons
    
    def break_bond(self, other_atom):
        # 断开与另一个原子的键
        self.bonds = [bond for bond in self.bonds if bond[0] != other_atom]
        other_atom.bonds = [bond for bond in other_atom.bonds if bond[0] != self]
        
        # 有几率电子返回 (离子键断裂时)
        if random.random() < 0.5:
            if self.charge < 0 and other_atom.charge > 0:  # 自己是负离子，对方是正离子
                self.electrons -= 1
                other_atom.electrons += 1
            elif self.charge > 0 and other_atom.charge < 0:  # 自己是正离子，对方是负离子
                self.electrons += 1
                other_atom.electrons -= 1
    
    def draw(self, surface, offset_x, offset_y, zoom):
        # 计算屏幕坐标
        screen_x = (self.x + offset_x) * zoom
        screen_y = (self.y + offset_y) * zoom
        radius = self.radius * zoom
        
        # 绘制原子
        pygame.draw.circle(surface, self.color, (int(screen_x), int(screen_y)), int(radius))
        
        # 如果被选中，绘制选中效果
        if self.selected:
            pygame.draw.circle(surface, GREEN, (int(screen_x), int(screen_y)), int(radius + 5), 2)
        
        # 绘制电子轨道
        for i in range(self.electrons):  # 使用实际电子数而不是价电子数
            angle = 2 * math.pi * i / max(1, self.electrons) + pygame.time.get_ticks() * 0.001
            orbit_radius = radius * 1.5
            electron_x = screen_x + orbit_radius * math.cos(angle)
            electron_y = screen_y + orbit_radius * math.sin(angle)
            pygame.draw.circle(surface, WHITE, (int(electron_x), int(electron_y)), int(radius * 0.3))
        
        if show_atom_names:
            font = pygame.font.Font(None, int(radius * 1.2))
            name_surface = font.render(self.name, True, WHITE)
            name_rect = name_surface.get_rect(center=(screen_x, screen_y))
            surface.blit(name_surface, name_rect)
        if show_charge:
        # 绘制电荷标记
            charge_font = pygame.font.Font(None, int(radius * 0.8))
            charge_text = f"{'+' if self.charge > 0 else ''}{self.charge}" if self.charge != 0 else "0"
            charge_surface = charge_font.render(charge_text, True, RED if self.charge > 0 else BLUE)
            charge_rect = charge_surface.get_rect(midtop=(screen_x, screen_y + radius))
            surface.blit(charge_surface, charge_rect)
    
    def draw_bonds(self, surface, offset_x, offset_y, zoom):
        for other_atom, strength in self.bonds:
            # 计算两个原子在屏幕上的位置
            x1 = (self.x + offset_x) * zoom
            y1 = (self.y + offset_y) * zoom
            x2 = (other_atom.x + offset_x) * zoom
            y2 = (other_atom.y + offset_y) * zoom
            
            # 绘制键 (线)
            bond_width = max(1, int(3 * strength * zoom))
            
            # 根据键类型选择颜色 (离子键或共价键)
            electroneg_diff = abs(self.electronegativity - other_atom.electronegativity)
            if electroneg_diff > 1.7:  # 离子键
                bond_color = (255, 165, 0)  # 橙色
            else:  # 共价键
                bond_color = (min(255, 100 + int(155 * strength)), 
                              min(255, 100 + int(155 * strength)), 
                              min(255, 100 + int(155 * strength)))
            
            pygame.draw.line(surface, bond_color, (x1, y1), (x2, y2), bond_width)
    
    def to_dict(self):
        """将原子数据转换为字典，用于保存"""
        return {
            'x': self.x,
            'y': self.y,
            'type': self.type,
            'radius': self.radius,
            'color': self.color,
            'valence_electrons': self.valence_electrons,
            'mass': self.mass,
            'electronegativity': self.electronegativity,
            'vx': self.vx,
            'vy': self.vy,
            'charge': self.charge,
            'electrons': self.electrons,
            'name': self.name,
            'bond_capacity': self.bond_capacity,
            'bond_length': self.bond_length,
            'id': self.id
        }
    
    @classmethod
    def from_dict(cls, data):
        """从字典创建原子"""
        atom = cls(data['x'], data['y'], data['type'])
        atom.radius = data['radius']
        atom.color = data['color']
        atom.valence_electrons = data['valence_electrons']
        atom.mass = data['mass']
        atom.electronegativity = data.get('electronegativity', 1.0)
        atom.vx = data['vx']
        atom.vy = data['vy']
        atom.charge = data['charge']
        atom.electrons = data['electrons']
        atom.name = data['name']
        atom.bond_capacity = data['bond_capacity']
        atom.bond_length = data['bond_length']
        atom.id = data['id']
        return atom

class QuadTree:
    def __init__(self, boundary, capacity):
        self.boundary = boundary  # 边界矩形，包含x, y, width, height
        self.capacity = capacity  # 每个节点的最大原子数量
        self.atoms = []  # 存储在当前节点的原子
        self.divided = False  # 是否已经划分子节点
        self.nw = self.ne = self.sw = self.se = None  # 四个子节点

    def subdivide(self):
        # 划分四个子节点
        x, y, w, h = self.boundary
        nw_bound = (x, y, w / 2, h / 2)
        ne_bound = (x + w / 2, y, w / 2, h / 2)
        sw_bound = (x, y + h / 2, w / 2, h / 2)
        se_bound = (x + w / 2, y + h / 2, w / 2, h / 2)
        self.nw = QuadTree(nw_bound, self.capacity)
        self.ne = QuadTree(ne_bound, self.capacity)
        self.sw = QuadTree(sw_bound, self.capacity)
        self.se = QuadTree(se_bound, self.capacity)
        self.divided = True

    def insert(self, atom):
        # 检查原子是否在当前节点的边界内
        if not (self.boundary[0] <= atom.x < self.boundary[0] + self.boundary[2] and
                self.boundary[1] <= atom.y < self.boundary[1] + self.boundary[3]):
            return False

        # 如果当前节点未满且未划分，直接插入
        if len(self.atoms) < self.capacity and not self.divided:
            self.atoms.append(atom)
            return True

        # 如果已划分，尝试插入到子节点
        if not self.divided:
            self.subdivide()
        return (self.nw.insert(atom) or self.ne.insert(atom) or
                self.sw.insert(atom) or self.se.insert(atom))

    def query_range(self, range_rect):
        # 查询范围内的原子
        found_atoms = []
        if not self.intersects(range_rect):
            return found_atoms

        for atom in self.atoms:
            if (range_rect[0] <= atom.x < range_rect[0] + range_rect[2] and
                    range_rect[1] <= atom.y < range_rect[1] + range_rect[3]):
                found_atoms.append(atom)

        if self.divided:
            found_atoms.extend(self.nw.query_range(range_rect))
            found_atoms.extend(self.ne.query_range(range_rect))
            found_atoms.extend(self.sw.query_range(range_rect))
            found_atoms.extend(self.se.query_range(range_rect))

        return found_atoms

    def intersects(self, range_rect):
        # 检查当前节点边界是否与查询范围相交
        return (self.boundary[0] < range_rect[0] + range_rect[2] and
                self.boundary[0] + self.boundary[2] > range_rect[0] and
                self.boundary[1] < range_rect[1] + range_rect[3] and
                self.boundary[1] + self.boundary[3] > range_rect[1])

# 使用四叉树优化碰撞检测和键形成检测
def process_interactions(atoms):
    # 创建四叉树
    quad_tree = QuadTree((0, 0, WIDTH * 10, HEIGHT * 10), 4)  # 假设世界范围是宽度和高度的10倍
    for atom in atoms:
        quad_tree.insert(atom)

    for atom in atoms:
        # 查询当前原子附近的原子
        query_rect = (atom.x - atom.radius * 2, atom.y - atom.radius * 2,
                      atom.radius * 4, atom.radius * 4)  # 查询范围为原子周围两倍半径的矩形
        nearby_atoms = quad_tree.query_range(query_rect)

        for other_atom in nearby_atoms:
            if atom == other_atom:
                continue

            # 计算距离 (考虑周期性边界)
            dx = other_atom.x - atom.x
            dy = other_atom.y - atom.y

            # 考虑周期性边界
            if dx > atom.world_width / 2:
                dx -= atom.world_width
            elif dx < -atom.world_width / 2:
                dx += atom.world_width
            if dy > atom.world_height / 2:
                dy -= atom.world_height
            elif dy < -atom.world_height / 2:
                dy += atom.world_height

            distance = math.sqrt(dx * dx + dy * dy)

            # 碰撞检测
            if distance < atom.radius + other_atom.radius:
                # 弹性碰撞响应
                nx = dx / distance
                ny = dy / distance
                p = 2 * (atom.vx * nx + atom.vy * ny - other_atom.vx * nx - other_atom.vy * ny) / (atom.mass + other_atom.mass)

                atom.vx = atom.vx - p * other_atom.mass * nx
                atom.vy = atom.vy - p * other_atom.mass * ny
                other_atom.vx = other_atom.vx + p * atom.mass * nx
                other_atom.vy = other_atom.vy + p * atom.mass * ny

                # 稍微分开原子防止粘连
                overlap = (atom.radius + other_atom.radius - distance) / 2
                atom.x -= overlap * nx
                atom.y -= overlap * ny
                other_atom.x += overlap * nx
                other_atom.y += overlap * ny

            # 键形成/断裂条件
            current_distance = distance
            bond_length = atom.bond_length + other_atom.bond_length

            # 检查是否已经形成键
            is_bonded = other_atom in [bond[0] for bond in atom.bonds]

            # 键形成/断裂逻辑
            if not is_bonded and current_distance < bond_length * 1.2:
                # 尝试形成键 (基于电负性和距离)
                if random.random() < 0.05:  # 有5%几率尝试形成键
                    atom.form_bond(other_atom)
            elif is_bonded and current_distance > bond_length * 1.5:
                # 键断裂条件
                if random.random() < 0.02:  # 有2%几率断裂
                    atom.break_bond(other_atom)
                    
def save_state(atoms, filename='atom_sim_state.json'):
    """保存当前状态到文件"""
    state = {
        'atoms': [atom.to_dict() for atom in atoms]
    }
    
    try:
        with open(filename, 'w') as f:
            json.dump(state, f)
        return True
    except Exception as e:
        print(f"保存状态失败: {e}")
        return False

def load_state(filename='atom_sim_state.json'):
    """从文件加载状态"""
    try:
        with open(filename, 'r') as f:
            state = json.load(f)
        
        atoms = []
        # 首先创建所有原子
        for atom_data in state['atoms']:
            atoms.append(Atom.from_dict(atom_data))
        
        # 然后重建键关系
        for atom in atoms:
            # 我们需要找到键对应的原子
            for bond_data in atom.bonds:
                other_atom_id = bond_data[0].id  # 假设我们保存了键的id
                # 找到对应的原子
                for other_atom in atoms:
                    if other_atom.id == other_atom_id:
                        atom.bonds.append((other_atom, bond_data[1]))
                        break
        
        return atoms
    except Exception as e:
        print(f"加载状态失败: {e}")
        return None

def get_atom_info(atom):
    """获取原子的详细信息"""
    info = [
        f"名称: {atom.name}",
        f"类型: {atom.type}",
        f"质量: {atom.mass:.2f}",
        f"电荷: {atom.charge}",
        f"价电子: {atom.valence_electrons}",
        f"实际电子: {atom.electrons}",
        f"电负性: {atom.electronegativity:.2f}",
        f"位置: ({atom.x:.1f}, {atom.y:.1f})",
        f"速度: ({atom.vx:.2f}, {atom.vy:.2f})",
        f"半径: {atom.radius:.1f}",
        f"键数: {len(atom.bonds)}/{atom.bond_capacity}"
    ]
    return info

def center_camera_on_atom(atom, offset_x, offset_y, zoom):
    """将摄像头中心对准指定原子"""
    # 计算原子在屏幕上的理想位置 (中心)
    target_screen_x = WIDTH / 2
    target_screen_y = HEIGHT / 2
    
    # 计算需要的偏移量
    new_offset_x = (target_screen_x / zoom) - atom.x
    new_offset_y = (target_screen_y / zoom) - atom.y
    
    return new_offset_x, new_offset_y

# 初始化原子
atoms = []
for _ in range(50):  # 初始原子数量
    x = random.randint(0, WIDTH * 10)
    y = random.randint(0, HEIGHT * 10)
    atoms.append(Atom(x, y))

# 添加一些特定类型的原子
for _ in range(10):
    x = random.randint(0, WIDTH * 10)
    y = random.randint(0, HEIGHT * 10)
    atoms.append(Atom(x, y, "金属"))

for _ in range(10):
    x = random.randint(0, WIDTH * 10)
    y = random.randint(0, HEIGHT * 10)
    atoms.append(Atom(x, y, "非金属"))

# 视角控制
offset_x = -WIDTH * 4.5
offset_y = -HEIGHT * 4.5
zoom = 0.1
dragging = False
last_mouse_pos = (0, 0)

# 主游戏循环
clock = pygame.time.Clock()
running = True
spawn_timer = 0
show_bonds = True
show_info = True
selected_atom = None
info_panel_visible = False

while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == MOUSEBUTTONDOWN:
            if event.button == 1:  # 左键
                # 检查是否点击了原子
                mouse_x, mouse_y = event.pos
                world_x = (mouse_x / zoom) - offset_x
                world_y = (mouse_y / zoom) - offset_y
                
                clicked_atom = None
                for atom in atoms:
                    dx = atom.x - world_x
                    dy = atom.y - world_y
                    distance = math.sqrt(dx*dx + dy*dy)
                    if distance < atom.radius:
                        clicked_atom = atom
                        break
                
                if clicked_atom:
                    # 取消之前选中的原子
                    if selected_atom:
                        selected_atom.selected = False
                    
                    # 选中新原子
                    selected_atom = clicked_atom
                    selected_atom.selected = True
                    info_panel_visible = True
                else:
                    # 如果没有点击原子，开始拖动
                    dragging = True
                    last_mouse_pos = event.pos
                    # 隐藏信息面板
                    info_panel_visible = False
                    if selected_atom:
                        selected_atom.selected = False
                        selected_atom = None
            elif event.button == 4:  # 滚轮上滚
                zoom *= 1.1
            elif event.button == 5:  # 滚轮下滚
                zoom /= 1.1
        elif event.type == MOUSEBUTTONUP:
            if event.button == 1:  # 左键释放
                dragging = False
        elif event.type == MOUSEMOTION:
            if dragging:
                dx = event.pos[0] - last_mouse_pos[0]
                dy = event.pos[1] - last_mouse_pos[1]
                offset_x += dx / zoom
                offset_y += dy / zoom
                last_mouse_pos = event.pos
        elif event.type == KEYDOWN:
            if event.key == K_SPACE:
                # 空格键添加新原子
                x = (WIDTH/2 - offset_x) / zoom  # 屏幕中心的世界坐标
                y = (HEIGHT/2 - offset_y) / zoom
                atoms.append(Atom(x, y))
            elif event.key == K_b:
                # B键切换键的显示
                show_bonds = not show_bonds
            elif event.key == K_i:
                # I键切换信息显示
                show_info = not show_info
            elif event.key == K_c:
                # C键清除所有原子
                atoms = []
                selected_atom = None
                info_panel_visible = False
            elif event.key == K_s:
                # S键保存状态
                if save_state(atoms):
                    print("状态已保存")
            elif event.key == K_l:
                # L键加载状态
                loaded_atoms = load_state()
                if loaded_atoms is not None:
                    atoms = loaded_atoms
                    selected_atom = None
                    info_panel_visible = False
                    print("状态已加载")
            elif event.key == K_f:
                # F键聚焦随机原子
                if atoms:
                    random_atom = random.choice(atoms)
                    offset_x, offset_y = center_camera_on_atom(random_atom, offset_x, offset_y, zoom)
            elif event.key == K_n:  # 按下 N 键
                show_atom_names = not show_atom_names  # 切换原子名字的显示状态
            elif event.key == K_t:  # 按下 T 键
                show_charge = not show_charge  # 切换原子电荷的显示状态
    
    # 随机生成新原子 - 扩大生成范围到300单位
    spawn_timer += 1
    if spawn_timer >= 60 and len(atoms) < 500:  # 限制最大原子数量为500
        spawn_timer = 0
        if atoms:  # 如果已经有原子存在
            # 随机选择一个已存在的原子作为中心原子
            center_atom = random.choice(atoms)
            # 在中心原子的附近生成新原子 (范围扩大到300单位)
            x = center_atom.x + random.uniform(-300, 300)
            y = center_atom.y + random.uniform(-300, 300)
        else:
            # 如果还没有原子，随机生成一个初始原子
            x = random.randint(0, WIDTH * 10)
            y = random.randint(0, HEIGHT * 10)
        atoms.append(Atom(x, y))
    
    # 更新原子
    for atom in atoms[:]:
        atom.update(atoms)
    
    # 处理原子间相互作用
    process_interactions(atoms)
    
    # 绘制
    screen.fill(BLACK)
    
    # 绘制所有键
    if show_bonds:
        for atom in atoms:
            atom.draw_bonds(screen, offset_x, offset_y, zoom)
    
    # 绘制所有原子
    for atom in atoms:
        atom.draw(screen, offset_x, offset_y, zoom)
    
    # 显示信息
    if show_info:
        font = pygame.font.Font("font.otf", 16)
        info_texts = [
            f"原子数量: {len(atoms)} | 缩放: {zoom:.2f}x",
            "控制: 拖动视角(鼠标左键) | 缩放(鼠标滚轮) | 添加原子(空格键)",
            "显示/隐藏键(B) | 显示/隐藏信息(I) | 清除所有原子(C)",
            "保存状态(S) | 加载状态(L) | 聚焦随机原子(F) | 点击原子查看详情",
            "显示/隐藏原子名字(N) | 显示/隐藏原子电荷(T)"
        ]
        
        for i, text in enumerate(info_texts):
            text_surface = font.render(text, True, WHITE)
            screen.blit(text_surface, (10, 10 + i * 25))
    
    # 显示选中的原子信息
    if selected_atom and info_panel_visible:
        info = get_atom_info(selected_atom)
        
        # 计算面板位置和大小
        panel_width = 300  # 加宽以显示更多信息
        panel_height = len(info) * 25 + 20
        panel_x = WIDTH - panel_width - 10
        panel_y = 10
        
        # 绘制面板背景
        pygame.draw.rect(screen, (50, 50, 80), (panel_x, panel_y, panel_width, panel_height))
        pygame.draw.rect(screen, GREEN, (panel_x, panel_y, panel_width, panel_height), 2)
        
        # 绘制原子信息
        font = pygame.font.Font("font.otf", 22)
        for i, line in enumerate(info):
            text_surface = font.render(line, True, WHITE)
            screen.blit(text_surface, (panel_x + 10, panel_y + 10 + i * 25))
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()