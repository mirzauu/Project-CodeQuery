import logging
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

import hashlib
import logging
import time
from typing import Dict, Optional

from neo4j import GraphDatabase
from sqlalchemy.orm import Session

import logging
import math
import os
import warnings
from collections import Counter, defaultdict, namedtuple
from pathlib import Path

import networkx as nx
from grep_ast import TreeContext, filename_to_lang
from pygments.lexers import guess_lexer_for_filename
from pygments.token import Token
from pygments.util import ClassNotFound
from tqdm import tqdm
from tree_sitter_languages import get_language, get_parser



# tree_sitter is throwing a FutureWarning
warnings.simplefilter("ignore", category=FutureWarning)
Tag = namedtuple("Tag", "rel_fname fname line end_line name kind type".split())


class RepoMap:
    # Parsing logic adapted from aider (https://github.com/paul-gauthier/aider)
    # Modified and customized for potpie's parsing needs with detailed tags, relationship tracking etc

    def __init__(
        self,
        map_tokens=1024,
        root=None,
        main_model=None,
        io=None,
        repo_content_prefix=None,
        verbose=False,
        max_context_window=None,
        map_mul_no_files=8,
    ):
        self.io = io
        self.verbose = verbose

        if not root:
            root = os.getcwd()
        self.root = root

        self.max_map_tokens = map_tokens
        self.map_mul_no_files = map_mul_no_files
        self.max_context_window = max_context_window

        self.repo_content_prefix = repo_content_prefix
        self.parse_helper = None

    def get_repo_map(
        self, chat_files, other_files, mentioned_fnames=None, mentioned_idents=None
    ):
        if self.max_map_tokens <= 0:
            return
        if not other_files:
            return
        if not mentioned_fnames:
            mentioned_fnames = set()
        if not mentioned_idents:
            mentioned_idents = set()

        max_map_tokens = self.max_map_tokens

        # With no files in the chat, give a bigger view of the entire repo
        padding = 4096
        if max_map_tokens and self.max_context_window:
            target = min(
                max_map_tokens * self.map_mul_no_files,
                self.max_context_window - padding,
            )
        else:
            target = 0
        if not chat_files and self.max_context_window and target > 0:
            max_map_tokens = target

        try:
            files_listing = self.get_ranked_tags_map(
                chat_files,
                other_files,
                max_map_tokens,
                mentioned_fnames,
                mentioned_idents,
            )
        except RecursionError:
            self.io.tool_error("Disabling repo map, git repo too large?")
            self.max_map_tokens = 0
            return

        if not files_listing:
            return

        num_tokens = self.token_count(files_listing)
        if self.verbose:
            self.io.tool_output(f"Repo-map: {num_tokens / 1024:.1f} k-tokens")

        if chat_files:
            other = "other "
        else:
            other = ""

        if self.repo_content_prefix:
            repo_content = self.repo_content_prefix.format(other=other)
        else:
            repo_content = ""

        repo_content += files_listing

        return repo_content

    def get_rel_fname(self, fname):
        return os.path.relpath(fname, self.root)

    def split_path(self, path):
        path = os.path.relpath(path, self.root)
        return [path + ":"]

    def save_tags_cache(self):
        pass

    def get_mtime(self, fname):
        print("get_mtime")
        try:
            return os.path.getmtime(fname)
        except FileNotFoundError:
            self.io.tool_error(f"File not found error: {fname}")

    def get_tags(self, fname, rel_fname):
        print("get_tag",fname,rel_fname)
        # Check if the file is in the cache and if the modification time has not changed
        file_mtime = self.get_mtime(fname)
        print("file_mtime",file_mtime)
        if file_mtime is None:
            return []

        data = list(self.get_tags_raw(fname, rel_fname))
  
        return data

    def get_tags_raw(self, fname, rel_fname):
        lang = filename_to_lang(fname)
        print(f"[DEBUG] Lang for {fname}: {lang}")
        if not lang:
            return

        language = get_language(lang)
        parser = get_parser(lang)
        print(f"[DEBUG] Language: {language}, Parser: {parser}")

        query_scm = get_scm_fname(lang)
     

        if not query_scm.exists():
            return
        query_scm = query_scm.read_text()

        code = self.io.read_text(fname)
        
        if not code:
            print("not code")
            return
        tree = parser.parse(bytes(code, "utf-8"))

        # Run the tags queries
        query = language.query(query_scm)
        captures = query.captures(tree.root_node)
        captures = list(captures)
        print(f"[DEBUG] {len(captures)} captures found")
        saw = set()

        for node, tag in captures:
            node_text = node.text.decode("utf-8")

            if tag.startswith("name.definition."):
                kind = "def"
                type = tag.split(".")[-1]

            elif tag.startswith("name.reference."):
                kind = "ref"
                type = tag.split(".")[-1]

            else:
                continue

            saw.add(kind)

            # Enhanced node text extraction for Java methods
            if lang == "java" and type == "method":
                # Handle method calls with object references (e.g., productService.listAllProducts())
                parent = node.parent
                if parent and parent.type == "method_invocation":
                    object_node = parent.child_by_field_name("object")
                    if object_node:
                        node_text = f"{object_node.text.decode('utf-8')}.{node_text}"

            result = Tag(
                rel_fname=rel_fname,
                fname=fname,
                name=node_text,
                kind=kind,
                line=node.start_point[0],
                end_line=node.end_point[0],
                type=type,
            )

            yield result

        if "ref" in saw:
            return
        if "def" not in saw:
            return

        try:
            lexer = guess_lexer_for_filename(fname, code)
        except ClassNotFound:
            return

        tokens = list(lexer.get_tokens(code))
        tokens = [token[1] for token in tokens if token[0] in Token.Name]

        for token in tokens:
            yield Tag(
                rel_fname=rel_fname,
                fname=fname,
                name=token,
                kind="ref",
                line=-1,
                end_line=-1,
                type="unknown",
            )

    @staticmethod
    def get_tags_from_code(fname, code):
        lang = filename_to_lang(fname)
        if not lang:
            return

        language = get_language(lang)
        parser = get_parser(lang)

        query_scm = get_scm_fname(lang)
        if not query_scm.exists():
            return
        query_scm = query_scm.read_text()

        if not code:
            return
        tree = parser.parse(bytes(code, "utf-8"))

        # Run the tags queries
        query = language.query(query_scm)
        captures = query.captures(tree.root_node)

        captures = list(captures)

        saw = set()
        for node, tag in captures:
            if tag.startswith("name.definition."):
                kind = "def"
                type = tag.split(".")[-1]  #
            elif tag.startswith("name.reference."):
                kind = "ref"
                type = tag.split(".")[-1]  #
            else:
                continue

            saw.add(kind)

            result = Tag(
                rel_fname=fname,
                fname=fname,
                name=node.text.decode("utf-8"),
                kind=kind,
                line=node.start_point[0],
                end_line=node.end_point[0],
                type=type,
            )

            yield result

        if "ref" in saw:
            return
        if "def" not in saw:
            return

        # We saw defs, without any refs
        # Some tags files only provide defs (cpp, for example)
        # Use pygments to backfill refs

        try:
            lexer = guess_lexer_for_filename(fname, code)
        except ClassNotFound:
            return

        tokens = list(lexer.get_tokens(code))
        tokens = [token[1] for token in tokens if token[0] in Token.Name]

        for token in tokens:
            yield Tag(
                rel_fname=fname,
                fname=fname,
                name=token,
                kind="ref",
                line=-1,
                end_line=-1,
                type="unknown",
            )

    def get_ranked_tags(
        self, chat_fnames, other_fnames, mentioned_fnames, mentioned_idents
    ):
        defines = defaultdict(set)
        references = defaultdict(list)
        definitions = defaultdict(set)

        personalization = dict()

        fnames = set(chat_fnames).union(set(other_fnames))
        chat_rel_fnames = set()

        fnames = sorted(fnames)

        # Default personalization for unspecified files is 1/num_nodes
        # https://networkx.org/documentation/stable/_modules/networkx/algorithms/link_analysis/pagerank_alg.html#pagerank
        personalize = 100 / len(fnames)

        fnames = tqdm(fnames)

        for fname in fnames:
            if not Path(fname).is_file():
                if fname not in self.warned_files:
                    if Path(fname).exists():
                        self.io.tool_error(
                            f"Repo-map can't include {fname}, it is not a normal file"
                        )
                    else:
                        self.io.tool_error(
                            f"Repo-map can't include {fname}, it no longer exists"
                        )

                self.warned_files.add(fname)
                continue

            # dump(fname)
            rel_fname = self.get_rel_fname(fname)

            if fname in chat_fnames:
                personalization[rel_fname] = personalize
                chat_rel_fnames.add(rel_fname)

            if rel_fname in mentioned_fnames:
                personalization[rel_fname] = personalize

            tags = list(self.get_tags(fname, rel_fname))
            if tags is None:
                continue

            for tag in tags:
                if tag.kind == "def":
                    defines[tag.name].add(rel_fname)
                    key = (rel_fname, tag.name)
                    definitions[key].add(tag)

                if tag.kind == "ref":
                    references[tag.name].append(rel_fname)

        ##
        # dump(defines)
        # dump(references)
        # dump(personalization)

        if not references:
            references = dict((k, list(v)) for k, v in defines.items())

        idents = set(defines.keys()).intersection(set(references.keys()))

        G = nx.MultiDiGraph()

        for ident in idents:
            definers = defines[ident]
            if ident in mentioned_idents:
                mul = 10
            elif ident.startswith("_"):
                mul = 0.1
            else:
                mul = 1

            for referencer, num_refs in Counter(references[ident]).items():
                for definer in definers:
                    # dump(referencer, definer, num_refs, mul)
                    # if referencer == definer:
                    #    continue

                    # scale down so high freq (low value) mentions don't dominate
                    num_refs = math.sqrt(num_refs)

                    G.add_edge(referencer, definer, weight=mul * num_refs, ident=ident)

        if not references:
            pass

        if personalization:
            pers_args = dict(personalization=personalization, dangling=personalization)
        else:
            pers_args = dict()

        try:
            ranked = nx.pagerank(G, weight="weight", **pers_args)
        except ZeroDivisionError:
            return []

        # distribute the rank from each source node, across all of its out edges
        ranked_definitions = defaultdict(float)
        for src in G.nodes:
            src_rank = ranked[src]
            total_weight = sum(
                data["weight"] for _src, _dst, data in G.out_edges(src, data=True)
            )
            # dump(src, src_rank, total_weight)
            for _src, dst, data in G.out_edges(src, data=True):
                data["rank"] = src_rank * data["weight"] / total_weight
                ident = data["ident"]
                ranked_definitions[(dst, ident)] += data["rank"]

        ranked_tags = []
        ranked_definitions = sorted(
            ranked_definitions.items(), reverse=True, key=lambda x: x[1]
        )

        # dump(ranked_definitions)

        for (fname, ident), rank in ranked_definitions:
            if fname in chat_rel_fnames:
                continue
            ranked_tags += list(definitions.get((fname, ident), []))

        rel_other_fnames_without_tags = set(
            self.get_rel_fname(fname) for fname in other_fnames
        )

        fnames_already_included = set(rt[0] for rt in ranked_tags)

        top_rank = sorted(
            [(rank, node) for (node, rank) in ranked.items()], reverse=True
        )
        for rank, fname in top_rank:
            if fname in rel_other_fnames_without_tags:
                rel_other_fnames_without_tags.remove(fname)
            if fname not in fnames_already_included:
                ranked_tags.append((fname,))

        for fname in rel_other_fnames_without_tags:
            ranked_tags.append((fname,))

        return ranked_tags

    def get_ranked_tags_map(
        self,
        chat_fnames,
        other_fnames=None,
        max_map_tokens=None,
        mentioned_fnames=None,
        mentioned_idents=None,
    ):
        if not other_fnames:
            other_fnames = list()
        if not max_map_tokens:
            max_map_tokens = self.max_map_tokens
        if not mentioned_fnames:
            mentioned_fnames = set()
        if not mentioned_idents:
            mentioned_idents = set()

        ranked_tags = self.get_ranked_tags(
            chat_fnames, other_fnames, mentioned_fnames, mentioned_idents
        )

        num_tags = len(ranked_tags)
        lower_bound = 0
        upper_bound = num_tags
        best_tree = None
        best_tree_tokens = 0

        chat_rel_fnames = [self.get_rel_fname(fname) for fname in chat_fnames]

        # Guess a small starting number to help with giant repos
        middle = min(max_map_tokens // 25, num_tags)

        self.tree_cache = dict()

        while lower_bound <= upper_bound:
            tree = self.to_tree(ranked_tags[:middle], chat_rel_fnames)
            num_tokens = self.token_count(tree)

            if num_tokens < max_map_tokens and num_tokens > best_tree_tokens:
                best_tree = tree
                best_tree_tokens = num_tokens

            if num_tokens < max_map_tokens:
                lower_bound = middle + 1
            else:
                upper_bound = middle - 1

            middle = (lower_bound + upper_bound) // 2

        return best_tree

    tree_cache = dict()

    def render_tree(self, abs_fname, rel_fname, lois):
        key = (rel_fname, tuple(sorted(lois)))

        if key in self.tree_cache:
            return self.tree_cache[key]

        code = self.io.read_text(abs_fname) or ""
        if not code.endswith("\n"):
            code += "\n"

        context = TreeContext(
            rel_fname,
            code,
            color=False,
            line_number=False,
            child_context=False,
            last_line=False,
            margin=0,
            mark_lois=False,
            loi_pad=0,
            show_top_of_file_parent_scope=False,
        )

        for start, end in lois:
            context.add_lines_of_interest(range(start, end + 1))
        context.add_context()
        res = context.format()
        self.tree_cache[key] = res
        return res

    def create_relationship(
        G, source, target, relationship_type, seen_relationships, extra_data=None
    ):
        """Helper to create relationships with proper direction checking"""
        if source == target:
            return False

        # Determine correct direction based on node types
        source_data = G.nodes[source]
        target_data = G.nodes[target]

        # Prevent duplicate bidirectional relationships
        rel_key = (source, target, relationship_type)
        reverse_key = (target, source, relationship_type)

        if rel_key in seen_relationships or reverse_key in seen_relationships:
            return False

        # Only create relationship if we have right direction:
        # 1. Interface method implementations should point to interface declaration
        # 2. Method calls should point to method definitions
        # 3. Class references should point to class definitions
        valid_direction = False

        if relationship_type == "REFERENCES":
            # Implementation -> Interface
            if (
                source_data.get("type") == "FUNCTION"
                and target_data.get("type") == "FUNCTION"
                and "Impl" in source
            ):  # Implementation class
                valid_direction = True

            # Caller -> Callee
            elif source_data.get("type") == "FUNCTION":
                valid_direction = True

            # Class Usage -> Class Definition
            elif target_data.get("type") == "CLASS":
                valid_direction = True

        if valid_direction:
            G.add_edge(source, target, type=relationship_type, **(extra_data or {}))
            seen_relationships.add(rel_key)
            return True

        return False
    
    def is_text_file(self, file_path):
        def open_text_file(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    f.read(1024)
                return True
            except UnicodeDecodeError:
                return False

        ext = file_path.split(".")[-1]
        exclude_extensions = [
            "png",
            "jpg",
            "jpeg",
            "gif",
            "bmp",
            "tiff",
            "webp",
            "ico",
            "svg",
            "mp4",
            "avi",
            "mov",
            "wmv",
            "flv",
            "ipynb",
        ]
        include_extensions = [
            "py",
            "js",
            "ts",
            "c",
            "cs",
            "cpp",
            "el",
            "ex",
            "exs",
            "elm",
            "go",
            "java",
            "ml",
            "mli",
            "php",
            "ql",
            "rb",
            "rs",
            "md",
            "txt",
            "json",
            "yaml",
            "yml",
            "toml",
            "ini",
            "cfg",
            "conf",
            "xml",
            "html",
            "css",
            "sh",
            "md",
            "mdx",
            "xsq",
            "proto",
        ]
        if ext in exclude_extensions:
            return False
        elif ext in include_extensions or open_text_file(file_path):
            return True
        else:
            return False
        
    def create_graph(self, repo_dir):
        print("start creating graph")
        G = nx.MultiDiGraph()
        defines = defaultdict(set)
        references = defaultdict(set)
        seen_relationships = set()
        print(G,defines,references,seen_relationships,repo_dir)
        for root, dirs, files in os.walk(repo_dir):
            if any(part.startswith(".") for part in root.split(os.sep)):
                continue

            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, repo_dir)
                print("relpath",rel_path)
                if not self.is_text_file(file_path):
                    continue

                logging.info(f"\nProcessing file: {rel_path}")

                # Add file node
                file_node_name = rel_path
                if not G.has_node(file_node_name):
                    print("g")
                    G.add_node(
                        file_node_name,
                        file=rel_path,
                        type="FILE",
                        text=self.io.read_text(file_path) or "",
                        line=0,
                        end_line=0,
                        name=rel_path.split("/")[-1],
                    )
                    

                current_class = None
                current_method = None

                # Process all tags in file
                tags = self.get_tags(file_path, rel_path)
                
                for tag in tags:
                    if tag.kind == "def":
                        print("tags",tags)
                        if tag.type == "class":
                            node_type = "CLASS"
                            current_class = tag.name
                            current_method = None
                        elif tag.type == "interface":
                            node_type = "INTERFACE"
                            current_class = tag.name
                            current_method = None
                        elif tag.type in ["method", "function"]:
                            node_type = "FUNCTION"
                            current_method = tag.name
                        else:
                            continue

                        # Create fully qualified node name
                        if current_class:
                            node_name = f"{rel_path}:{current_class}.{tag.name}"
                        else:
                            node_name = f"{rel_path}:{tag.name}"

                        # Add node
                        if not G.has_node(node_name):
                            G.add_node(
                                node_name,
                                file=rel_path,
                                line=tag.line,
                                end_line=tag.end_line,
                                type=node_type,
                                name=tag.name,
                                class_name=current_class,
                            )

                            # Add CONTAINS relationship from file
                            rel_key = (file_node_name, node_name, "CONTAINS")
                            if rel_key not in seen_relationships:
                                G.add_edge(
                                    file_node_name,
                                    node_name,
                                    type="CONTAINS",
                                    ident=tag.name,
                                )
                                seen_relationships.add(rel_key)
                                logging.info(f"Added edge: {file_node_name} -> {node_name} (CONTAINS)")

                        # Record definition
                        defines[tag.name].add(node_name)

                    elif tag.kind == "ref":
                        # Handle references
                        if current_class and current_method:
                            source = f"{rel_path}:{current_class}.{current_method}"
                        elif current_method:
                            source = f"{rel_path}:{current_method}"
                        else:
                            source = rel_path

                        references[tag.name].add(
                            (
                                source,
                                tag.line,
                                tag.end_line,
                                current_class,
                                current_method,
                            )
                        )
                        
        print("start relation")
        for ident, refs in references.items():
            target_nodes = defines.get(ident, set())
            print("Defines:", dict(defines))
            print("References:", dict(references))
            for source, line, end_line, src_class, src_method in refs:
                for target in target_nodes:
                    if source == target:
                        continue

                    if G.has_node(source) and G.has_node(target):
                        RepoMap.create_relationship(
                            G,
                            source,
                            target,
                            "REFERENCES",
                            seen_relationships,
                            {
                                "ident": ident,
                                "ref_line": line,
                                "end_ref_line": end_line,
                            },
                        )

        return G

    @staticmethod
    def get_language_for_file(file_path):
        # Map file extensions to tree-sitter languages
        extension = os.path.splitext(file_path)[1].lower()
        language_map = {
            ".py": get_language("python"),
            ".js": get_language("javascript"),
            ".ts": get_language("typescript"),
            ".c": get_language("c"),
            ".cs": get_language("c_sharp"),
            ".cpp": get_language("cpp"),
            ".el": get_language("elisp"),
            ".ex": get_language("elixir"),
            ".exs": get_language("elixir"),
            ".elm": get_language("elm"),
            ".go": get_language("go"),
            ".java": get_language("java"),
            ".ml": get_language("ocaml"),
            ".mli": get_language("ocaml"),
            ".php": get_language("php"),
            ".ql": get_language("ql"),
            ".rb": get_language("ruby"),
            ".rs": get_language("rust"),
        }
        return language_map.get(extension)

    @staticmethod
    def find_node_by_range(root_node, start_line, node_type):
        def traverse(node):
            if node.start_point[0] <= start_line and node.end_point[0] >= start_line:
                if node_type == "FUNCTION" and node.type in [
                    "function_definition",
                    "method",
                    "method_declaration",
                    "function",
                ]:
                    return node
                elif node_type in ["CLASS", "INTERFACE"] and node.type in [
                    "class_definition",
                    "interface",
                    "class",
                    "class_declaration",
                    "interface_declaration",
                ]:
                    return node
                for child in node.children:
                    result = traverse(child)
                    if result:
                        return result
            return None

        return traverse(root_node)

    def to_tree(self, tags, chat_rel_fnames):
        if not tags:
            return ""

        tags = [tag for tag in tags if tag[0] not in chat_rel_fnames]
        tags = sorted(tags)

        cur_fname = None
        cur_abs_fname = None
        lois = None
        output = ""

        # add a bogus tag at the end so we trip the this_fname != cur_fname...
        dummy_tag = (None,)
        for tag in tags + [dummy_tag]:
            this_rel_fname = tag[0]

            # ... here ... to output the final real entry in the list
            if this_rel_fname != cur_fname:
                if lois is not None:
                    output += "\n"
                    output += cur_fname + ":\n"
                    output += self.render_tree(cur_abs_fname, cur_fname, lois)
                    lois = None
                elif cur_fname:
                    output += "\n" + cur_fname + "\n"
                if type(tag) is Tag:
                    lois = []
                    cur_abs_fname = tag.fname
                cur_fname = this_rel_fname

            if lois is not None:
                lois.append((tag.line, tag.end_line))

        # truncate long lines, in case we get minified js or something else crazy
        output = "\n".join([line[:100] for line in output.splitlines()]) + "\n"

        return output


def get_scm_fname(lang):
    # Load the tags queries
    try:
        return Path(os.path.dirname(__file__)).joinpath(
            "queries", f"tree-sitter-{lang}-tags.scm"
        )
    except KeyError:
        return




class CodeGraphService:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password, db: Session):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.db = db

    @staticmethod
    def generate_node_id(path: str, user_id: str):
        # Concatenate path and signature
        combined_string = f"{user_id}:{path}"

        # Create a SHA-1 hash of the combined string
        hash_object = hashlib.md5()
        hash_object.update(combined_string.encode("utf-8"))

        # Get the hexadecimal representation of the hash
        node_id = hash_object.hexdigest()

        return node_id

    def close(self):
        self.driver.close()

    def create_and_store_graph(self, repo_dir, project_id, user_id):
        # Create the graph using RepoMap
        self.repo_map = RepoMap(
            root=repo_dir,
            verbose=True,
            main_model=SimpleTokenCounter(),
            io=SimpleIO(),
        )

        nx_graph = self.repo_map.create_graph(repo_dir)

        with self.driver.session() as session:
            start_time = time.time()
            node_count = nx_graph.number_of_nodes()
            logging.info(f"Creating {node_count} nodes")

            # Create specialized index for relationship queries
            session.run(
                """
                CREATE INDEX node_id_repo_idx IF NOT EXISTS
                FOR (n:NODE) ON (n.node_id, n.repoId)
            """
            )

            # Batch insert nodes
            batch_size = 1000
            for i in range(0, node_count, batch_size):
                batch_nodes = list(nx_graph.nodes(data=True))[i : i + batch_size]
                nodes_to_create = []

                for node_id, node_data in batch_nodes:
                    # Get the node type and ensure it's one of our expected types
                    node_type = node_data.get("type", "UNKNOWN")
                    if node_type == "UNKNOWN":
                        continue
                    # Initialize labels with NODE
                    labels = ["NODE"]

                    # Add specific type label if it's a valid type
                    if node_type in ["FILE", "CLASS", "FUNCTION", "INTERFACE"]:
                        labels.append(node_type)

                    # Prepare node data
                    processed_node = {
                        "name": node_data.get(
                            "name", node_id
                        ),  # Use node_id as fallback
                        "file_path": node_data.get("file", ""),
                        "start_line": node_data.get("line", -1),
                        "end_line": node_data.get("end_line", -1),
                        "repoId": project_id,
                        "node_id": CodeGraphService.generate_node_id(node_id, user_id),
                        "entityId": user_id,
                        "type": node_type,
                        "text": node_data.get("text", ""),
                        "labels": labels,
                    }

                    # Remove None values
                    processed_node = {
                        k: v for k, v in processed_node.items() if v is not None
                    }
                    nodes_to_create.append(processed_node)

                # Create nodes with labels
                session.run(
                    """
                    UNWIND $nodes AS node
                    CALL apoc.create.node(node.labels, node) YIELD node AS n
                    RETURN count(*) AS created_count
                    """,
                    nodes=nodes_to_create,
                )

            relationship_count = nx_graph.number_of_edges()
            logging.info(f"Creating {relationship_count} relationships")

            # Pre-calculate common relationship types to avoid dynamic relationship creation
            rel_types = set()
            for source, target, data in nx_graph.edges(data=True):
                rel_type = data.get("type", "REFERENCES")
                rel_types.add(rel_type)

            # Process relationships with huge batch size and type-specific queries
            batch_size = 1000

            for rel_type in rel_types:
                # Filter edges by relationship type
                type_edges = [
                    (s, t, d)
                    for s, t, d in nx_graph.edges(data=True)
                    if d.get("type", "REFERENCES") == rel_type
                ]

                logging.info(
                    f"Creating {len(type_edges)} relationships of type {rel_type}"
                )

                for i in range(0, len(type_edges), batch_size):
                    batch_edges = type_edges[i : i + batch_size]
                    edges_to_create = []

                    for source, target, data in batch_edges:
                        edges_to_create.append(
                            {
                                "source_id": CodeGraphService.generate_node_id(
                                    source, user_id
                                ),
                                "target_id": CodeGraphService.generate_node_id(
                                    target, user_id
                                ),
                                "repoId": project_id,
                            }
                        )

                    # Type-specific relationship creation in one transaction
                    query = f"""
                        UNWIND $edges AS edge
                        MATCH (source:NODE {{node_id: edge.source_id, repoId: edge.repoId}})
                        MATCH (target:NODE {{node_id: edge.target_id, repoId: edge.repoId}})
                        CREATE (source)-[r:{rel_type} {{repoId: edge.repoId}}]->(target)
                    """
                    session.run(query, edges=edges_to_create)

            end_time = time.time()
            logging.info(
                f"Time taken to create graph and search index: {end_time - start_time:.2f} seconds"
            )



    async def get_node_by_id(self, node_id: str, project_id: str) -> Optional[Dict]:
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (n:NODE {node_id: $node_id, repoId: $project_id})
                RETURN n
                """,
                node_id=node_id,
                project_id=project_id,
            )
            record = result.single()
            return dict(record["n"]) if record else None

    def query_graph(self, query):
        with self.driver.session() as session:
            result = session.run(query)
            return [record.data() for record in result]


class SimpleIO:
    def read_text(self, fname):
        try:
            with open(fname, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            logging.warning(f"Could not read {fname} as UTF-8. Skipping this file.")
            return ""

    def tool_error(self, message):
        logging.error(f"Error: {message}")

    def tool_output(self, message):
        logging.info(message)


class SimpleTokenCounter:
    def token_count(self, text):
        return len(text.split())

# Setup logging
logging.basicConfig(level=logging.INFO)
NEO4J_URI = "neo4j+s://0bf07ab5.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "1jO1KUcfWBwUKzz4U_87ZwVS4ozX2fEixs10iN6766E"  # Replace with your password





















import asyncio
import logging
import os
import re
from typing import Dict, List, Optional

import tiktoken
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session

# from app.core.config_provider import config_provider
# from app.modules.intelligence.provider.provider_service import (
#     ProviderService,
# )
# from app.modules.parsing.knowledge_graph.inference_schema import (
#     DocstringRequest,
#     DocstringResponse,
# )
# from app.modules.projects.projects_service import ProjectService
# from app.modules.search.search_service import SearchService

logger = logging.getLogger(__name__)

from typing import List, Optional,Any

from pydantic import BaseModel

import litellm
from litellm import litellm, AsyncOpenAI, acompletion
import instructor
import httpx
from pydantic import BaseModel

class DocstringRequest(BaseModel):
    node_id: str
    text: str


class DocstringNode(BaseModel):
    node_id: str
    docstring: str
    tags: Optional[List[str]] = []


class DocstringResponse(BaseModel):
    docstrings: List[DocstringNode]


class InferenceService:
    def __init__(self, db: Session, user_id: Optional[str] = "dummy"):
        # neo4j_config = config_provider.get_neo4j_config()
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
        )

        # self.provider_service = ProviderService(db, user_id)
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        # self.search_service = SearchService(db)
        # self.project_manager = ProjectService(db)
        self.parallel_requests = int(os.getenv("PARALLEL_REQUESTS", 50))

    def close(self):
        self.driver.close()

    def log_graph_stats(self, repo_id):
        query = """
        MATCH (n:NODE {repoId: $repo_id})
        OPTIONAL MATCH (n)-[r]-(m:NODE {repoId: $repo_id})
        RETURN
        COUNT(DISTINCT n) AS nodeCount,
        COUNT(DISTINCT r) AS relationshipCount
        """

        try:
            # Establish connection
            with self.driver.session() as session:
                # Execute the query
                result = session.run(query, repo_id=repo_id)
                record = result.single()

                if record:
                    node_count = record["nodeCount"]
                    relationship_count = record["relationshipCount"]

                    # Log the results
                    logger.info(
                        f"DEBUGNEO4J: Repo ID: {repo_id}, Nodes: {node_count}, Relationships: {relationship_count}"
                    )
                else:
                    logger.info(
                        f"DEBUGNEO4J: No data found for repository ID: {repo_id}"
                    )

        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")

    def num_tokens_from_string(self, string: str, model: str = "gpt-4") -> int:
        """Returns the number of tokens in a text string."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            logger.warning("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(string, disallowed_special=set()))

    def fetch_graph(self, repo_id: str) -> List[Dict]:
        batch_size = 500
        all_nodes = []
        with self.driver.session() as session:
            offset = 0
            while True:
                result = session.run(
                    "MATCH (n:NODE {repoId: $repo_id}) "
                    "RETURN n.node_id AS node_id, n.text AS text, n.file_path AS file_path, n.start_line AS start_line, n.end_line AS end_line, n.name AS name "
                    "SKIP $offset LIMIT $limit",
                    repo_id=repo_id,
                    offset=offset,
                    limit=batch_size,
                )
                batch = [dict(record) for record in result]
                if not batch:
                    break
                all_nodes.extend(batch)
                offset += batch_size
        logger.info(f"DEBUGNEO4J: Fetched {len(all_nodes)} nodes for repo {repo_id}")
        return all_nodes

    def get_entry_points(self, repo_id: str) -> List[str]:
        batch_size = 400  # Define the batch size
        all_entry_points = []
        with self.driver.session() as session:
            offset = 0
            while True:
                result = session.run(
                    f"""
                    MATCH (f:FUNCTION)
                    WHERE f.repoId = '{repo_id}'
                    AND NOT ()-[:CALLS]->(f)
                    AND (f)-[:CALLS]->()
                    RETURN f.node_id as node_id
                    SKIP $offset LIMIT $limit
                    """,
                    offset=offset,
                    limit=batch_size,
                )
                batch = result.data()
                if not batch:
                    break
                all_entry_points.extend([record["node_id"] for record in batch])
                offset += batch_size
        return all_entry_points

    def get_neighbours(self, node_id: str, repo_id: str):
        with self.driver.session() as session:
            batch_size = 400  # Define the batch size
            all_nodes_info = []
            offset = 0
            while True:
                result = session.run(
                    """
                    MATCH (start {node_id: $node_id, repoId: $repo_id})
                    OPTIONAL MATCH (start)-[:CALLS]->(direct_neighbour)
                    OPTIONAL MATCH (start)-[:CALLS]->()-[:CALLS*0..]->(indirect_neighbour)
                    WITH start, COLLECT(DISTINCT direct_neighbour) + COLLECT(DISTINCT indirect_neighbour) AS all_neighbours
                    UNWIND all_neighbours AS neighbour
                    WITH start, neighbour
                    WHERE neighbour IS NOT NULL AND neighbour <> start
                    RETURN DISTINCT neighbour.node_id AS node_id, neighbour.name AS function_name, labels(neighbour) AS labels
                    SKIP $offset LIMIT $limit
                    """,
                    node_id=node_id,
                    repo_id=repo_id,
                    offset=offset,
                    limit=batch_size,
                )
                batch = result.data()
                if not batch:
                    break
                all_nodes_info.extend(
                    [
                        record["node_id"]
                        for record in batch
                        if "FUNCTION" in record["labels"]
                    ]
                )
                offset += batch_size
            return all_nodes_info

    def get_entry_points_for_nodes(
        self, node_ids: List[str], repo_id: str
    ) -> Dict[str, List[str]]:
        with self.driver.session() as session:
            result = session.run(
                """
                UNWIND $node_ids AS nodeId
                MATCH (n:FUNCTION)
                WHERE n.node_id = nodeId and n.repoId = $repo_id
                OPTIONAL MATCH path = (entryPoint)-[*]->(n)
                WHERE NOT (entryPoint)<--()
                RETURN n.node_id AS input_node_id, collect(DISTINCT entryPoint.node_id) AS entry_point_node_ids

                """,
                node_ids=node_ids,
                repo_id=repo_id,
            )
            return {
                record["input_node_id"]: (
                    record["entry_point_node_ids"]
                    if len(record["entry_point_node_ids"]) > 0
                    else [record["input_node_id"]]
                )
                for record in result
            }

    def batch_nodes(
        self, nodes: List[Dict], max_tokens: int = 16000, model: str = "gpt-4"
    ) -> List[List[DocstringRequest]]:
        batches = []
        current_batch = []
        current_tokens = 0
        node_dict = {node["node_id"]: node for node in nodes}

        def replace_referenced_text(
            text: str, node_dict: Dict[str, Dict[str, str]]
        ) -> str:
            pattern = r"Code replaced for brevity\. See node_id ([a-f0-9]+)"
            regex = re.compile(pattern)

            def replace_match(match):
                node_id = match.group(1)
                if node_id in node_dict:
                    return "\n" + node_dict[node_id]["text"].split("\n", 1)[-1]
                return match.group(0)

            previous_text = None
            current_text = text

            while previous_text != current_text:
                previous_text = current_text
                current_text = regex.sub(replace_match, current_text)
            return current_text

        for node in nodes:
            if not node.get("text"):
                logger.warning(f"Node {node['node_id']} has no text. Skipping...")
                continue

            updated_text = replace_referenced_text(node["text"], node_dict)
            node_tokens = self.num_tokens_from_string(updated_text, model)

            if node_tokens > max_tokens:
                logger.warning(
                    f"Node {node['node_id']} - {node_tokens} tokens, has exceeded the max_tokens limit. Skipping..."
                )
                continue

            if current_tokens + node_tokens > max_tokens:
                if current_batch:  # Only append if there are items
                    batches.append(current_batch)
                current_batch = []
                current_tokens = 0

            current_batch.append(
                DocstringRequest(node_id=node["node_id"], text=updated_text)
            )
            current_tokens += node_tokens

        if current_batch:
            batches.append(current_batch)

        total_nodes = sum(len(batch) for batch in batches)
        logger.info(f"Batched {total_nodes} nodes into {len(batches)} batches")
        logger.info(f"Batch sizes: {[len(batch) for batch in batches]}")

        return batches

    async def generate_docstrings_for_entry_points(
        self,
        all_docstrings,
        entry_points_neighbors: Dict[str, List[str]],
    ) -> Dict[str, DocstringResponse]:
        docstring_lookup = {
            d.node_id: d.docstring for d in all_docstrings["docstrings"]
        }

        entry_point_batches = self.batch_entry_points(
            entry_points_neighbors, docstring_lookup
        )

        semaphore = asyncio.Semaphore(self.parallel_requests)

        async def process_batch(batch):
            async with semaphore:
                response = await self.generate_entry_point_response(batch)
                if isinstance(response, DocstringResponse):
                    return response
                else:
                    return await self.generate_docstrings_for_entry_points(
                        all_docstrings, entry_points_neighbors
                    )

        tasks = [process_batch(batch) for batch in entry_point_batches]
        results = await asyncio.gather(*tasks)

        updated_docstrings = DocstringResponse(docstrings=[])
        for result in results:
            updated_docstrings.docstrings.extend(result.docstrings)

        # Update all_docstrings with the new entry point docstrings
        for updated_docstring in updated_docstrings.docstrings:
            existing_index = next(
                (
                    i
                    for i, d in enumerate(all_docstrings["docstrings"])
                    if d.node_id == updated_docstring.node_id
                ),
                None,
            )
            if existing_index is not None:
                all_docstrings["docstrings"][existing_index] = updated_docstring
            else:
                all_docstrings["docstrings"].append(updated_docstring)

        return all_docstrings

    def batch_entry_points(
        self,
        entry_points_neighbors: Dict[str, List[str]],
        docstring_lookup: Dict[str, str],
        max_tokens: int = 16000,
        model: str = "gpt-4",
    ) -> List[List[Dict[str, str]]]:
        batches = []
        current_batch = []
        current_tokens = 0

        for entry_point, neighbors in entry_points_neighbors.items():
            entry_docstring = docstring_lookup.get(entry_point, "")
            neighbor_docstrings = [
                f"{neighbor}: {docstring_lookup.get(neighbor, '')}"
                for neighbor in neighbors
            ]
            flow_description = "\n".join(neighbor_docstrings)

            entry_point_data = {
                "node_id": entry_point,
                "entry_docstring": entry_docstring,
                "flow_description": entry_docstring + "\n" + flow_description,
            }

            entry_point_tokens = self.num_tokens_from_string(
                entry_docstring + flow_description, model
            )

            if entry_point_tokens > max_tokens:
                continue  # Skip entry points that exceed the max_tokens limit

            if current_tokens + entry_point_tokens > max_tokens:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0

            current_batch.append(entry_point_data)
            current_tokens += entry_point_tokens

        if current_batch:
            batches.append(current_batch)

        return batches
    async def call_llm_with_structured_output(
        self, messages: list, output_schema: type[BaseModel], config_type: str = "chat"
    ) -> Any:
        """Call LLM and parse the response into a structured output using a Pydantic model."""
        # Select the appropriate config
        import json
        from mistralai import Mistral

        api_key = "I8dVoJSO5XmpMUcyIQ0KRiGNfduJRCM8"
        model = "mistral-large-latest"

        client = Mistral(api_key=api_key)

# Initialize the client with your API key
        

        try:
            # Make the LLM request
            response = client.chat.complete(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
            )

            # Get the JSON string
            raw = response.choices[0].message.content
            logging.info(f"Raw LLM response: {raw}")

            # Parse the JSON string into Python list
            parsed_data = json.loads(raw)

            # ✅ Ensure it's wrapped in a dict with "docstrings" key
            if isinstance(parsed_data, list):
                parsed_data = {"docstrings": parsed_data}

            # ✅ Now validate the parsed_data against the Pydantic model
            validated_output = output_schema.model_validate(parsed_data)
            return validated_output

        except Exception as e:
            logging.error(f"LLM call with structured output failed: {e}")
            raise e
        
    async def generate_entry_point_response(
        self, batch: List[Dict[str, str]]
    ) -> DocstringResponse:
        prompt = """
        You are an expert software architect with deep knowledge of distributed systems and cloud-native applications. Your task is to analyze entry points and their function flows in a codebase.

        For each of the following entry points and their function flows, perform the following task:

        1. **Flow Summary**: Generate a concise yet comprehensive summary of the overall intent and purpose of the entry point and its flow. Follow these guidelines:
           - Start with a high-level overview of the entry point's purpose.
           - Detail the main steps or processes involved in the flow.
           - Highlight key interactions with external systems or services.
           - Specify ALL API paths, HTTP methods, topic names, database interactions, and critical function calls.
           - Identify any error handling or edge cases.
           - Conclude with the expected output or result of the flow.

        Remember, the summary should be technical enough for a senior developer to understand the code's functionality via similarity search, but concise enough to be quickly parsed. Aim for a balance between detail and brevity.

        Your response must be a valid JSON object containing a list of docstrings, where each docstring object has:
        - node_id: The ID of the entry point being documented
        - docstring: A comprehensive flow summary following the guidelines above
        - tags: A list of relevant tags based on the functionality (e.g., ["API", "DATABASE"] for endpoints that interact with a database)

        Here are the entry points and their flows:

        {entry_points}
        """

        entry_points_text = "\n\n".join(
            [
                f"Entry point: {entry_point['node_id']}\n"
                f"Flow:\n{entry_point['flow_description']}"
                f"Entry docstring:\n{entry_point['entry_docstring']}"
                for entry_point in batch
            ]
        )

        messages = [
            {
                "role": "system",
                "content": "You are an expert software architecture documentation assistant. You will analyze code flows and provide structured documentation in JSON format.",
            },
            {"role": "user", "content": prompt.format(entry_points=entry_points_text)},
        ]

        try:
            result = await self.call_llm_with_structured_output(
                messages=messages,
                output_schema=DocstringResponse,
                config_type="inference",
            )
            return result
        except Exception as e:
            logger.error(f"Entry point response generation failed: {e}")
            return DocstringResponse(docstrings=[])

    async def generate_docstrings(self, repo_id: str) -> Dict[str, DocstringResponse]:
        logger.info(
            f"DEBUGNEO4J: Function: {self.generate_docstrings.__name__}, Repo ID: {repo_id}"
        )
        self.log_graph_stats(repo_id)
        nodes = self.fetch_graph(repo_id)
        logger.info(
            f"DEBUGNEO4J: After fetch graph, Repo ID: {repo_id}, Nodes: {len(nodes)}"
        )
        self.log_graph_stats(repo_id)
        logger.info(
            f"Creating search indices for project {repo_id} with nodes count {len(nodes)}"
        )

        # Prepare a list of nodes for bulk insert
        nodes_to_index = [
            {
                "project_id": repo_id,
                "node_id": node["node_id"],
                "name": node.get("name", ""),
                "file_path": node.get("file_path", ""),
                "content": f"{node.get('name', '')} {node.get('file_path', '')}",
            }
            for node in nodes
            if node.get("file_path") not in {None, ""}
            and node.get("name") not in {None, ""}
        ]

        # Perform bulk insert
        # await self.search_service.bulk_create_search_indices(nodes_to_index)

        # logger.info(
        #     f"Project {repo_id}: Created search indices over {len(nodes_to_index)} nodes"
        # )

        # await self.search_service.commit_indices()
        # entry_points = self.get_entry_points(repo_id)
        # logger.info(
        #     f"DEBUGNEO4J: After get entry points, Repo ID: {repo_id}, Entry points: {len(entry_points)}"
        # )
        # self.log_graph_stats(repo_id)
        # entry_points_neighbors = {}
        # for entry_point in entry_points:
        #     neighbors = self.get_neighbours(entry_point, repo_id)
        #     entry_points_neighbors[entry_point] = neighbors

        # logger.info(
        #     f"DEBUGNEO4J: After get neighbours, Repo ID: {repo_id}, Entry points neighbors: {len(entry_points_neighbors)}"
        # )
        # self.log_graph_stats(repo_id)
        batches = self.batch_nodes(nodes)
        all_docstrings = {"docstrings": []}

        semaphore = asyncio.Semaphore(self.parallel_requests)

        async def process_batch(batch, batch_index: int):
            async with semaphore:
                logger.info(f"Processing batch {batch_index} for project {repo_id}")
                response = await self.generate_response(batch, repo_id)
                if not isinstance(response, DocstringResponse):
                    logger.warning(
                        f"Parsing project {repo_id}: Invalid response from LLM. Not an instance of DocstringResponse. Retrying..."
                    )
                    response = await self.generate_response(batch, repo_id)
                else:
                    self.update_neo4j_with_docstrings(repo_id, response)
                return response

        tasks = [process_batch(batch, i) for i, batch in enumerate(batches)]
        results = await asyncio.gather(*tasks)

        for result in results:
            if not isinstance(result, DocstringResponse):
                logger.error(
                    f"Project {repo_id}: Invalid response from during inference. Manually verify the project completion."
                )

        # updated_docstrings = await self.generate_docstrings_for_entry_points(
        #     all_docstrings, entry_points_neighbors
        # )
        updated_docstrings = all_docstrings
        return updated_docstrings

    async def generate_response(
        self, batch: List[DocstringRequest], repo_id: str
    ) -> DocstringResponse:
        base_prompt = """
        You are a senior software engineer specializing in code analysis and documentation. Your task is to generate concise docstrings and relevant tags for each code snippet in the provided `code_snippets`.

        **Instructions**:

        1. **Process Each Node**:
        - For every `node_id` in the input, analyze the associated code block.

        2. **Determine Code Type**:
        - Classify as **backend** or **frontend** using common patterns:
            - Backend: DB access, APIs, config, server logic.
            - Frontend: UI, state, events, styling.

        3. **Summarize Purpose**:
        - Write a brief (1–2 sentence) docstring summarizing what the code does and its role.

        4. **Assign Tags**:
        - Choose relevant tags from these:

        **Backend**: AUTH, DATABASE, API, UTILITY, PRODUCER, CONSUMER, EXTERNAL_SERVICE, CONFIGURATION  
        **Frontend**: UI_COMPONENT, FORM_HANDLING, STATE_MANAGEMENT, DATA_BINDING, ROUTING, EVENT_HANDLING, STYLING, MEDIA, ANIMATION, ACCESSIBILITY, DATA_FETCHING

        **Output Format**:
        Return a JSON list of objects with:
        - `node_id`: ID of the code node
        - `docstring`: Concise purpose
        - `tags`: List of tags

        Here are the code snippets:

        {code_snippets}
        """


        # Prepare the code snippets
        code_snippets = ""
        for request in batch:
            code_snippets += (
                f"node_id: {request.node_id} \n```\n{request.text}\n```\n\n "
            )
    
        messages = [
            {
                "role": "system",
                "content": "You are an expert software documentation assistant. You will analyze code and provide structured documentation in JSON format.",
            },
            {
                "role": "user",
                "content": base_prompt.format(code_snippets=code_snippets),
            },
        ]

        import time

        start_time = time.time()
        logger.info(f"Parsing project {repo_id}: Starting the inference process...")

        try:
            result = await self.call_llm_with_structured_output(
                messages=messages,
                output_schema=DocstringResponse,
                config_type="inference",
            )
        except Exception as e:
            logger.error(
                f"Parsing project {repo_id}: Inference request failed. Error: {str(e)}"
            )
            result = DocstringResponse(docstrings=[])

        end_time = time.time()
        logger.info(
            f"Parsing project {repo_id}: Inference request completed. Time Taken: {end_time - start_time} seconds"
        )
        return result

    def generate_embedding(self, text: str) -> List[float]:
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()

    def update_neo4j_with_docstrings(self, repo_id: str, docstrings: DocstringResponse):
        with self.driver.session() as session:
            batch_size = 300
            docstring_list = [
                {
                    "node_id": n.node_id,
                    "docstring": n.docstring,
                    "tags": n.tags,
                    "embedding": self.generate_embedding(n.docstring),
                }
                for n in docstrings.docstrings
            ]
            # project = self.project_manager.get_project_from_db_by_id_sync(repo_id)
            # repo_path = project.get("repo_path")
            is_local_repo = True 
            for i in range(0, len(docstring_list), batch_size):
                batch = docstring_list[i : i + batch_size]
                session.run(
                    """
                    UNWIND $batch AS item
                    MATCH (n:NODE {repoId: $repo_id, node_id: item.node_id})
                    SET n.docstring = item.docstring,
                        n.embedding = item.embedding,
                        n.tags = item.tags
                    """
                    + ("" if is_local_repo else "REMOVE n.text, n.signature"),
                    batch=batch,
                    repo_id=repo_id,
                )

    def create_vector_index(self):
        with self.driver.session() as session:
            session.run(
                """
                CREATE VECTOR INDEX docstring_embedding IF NOT EXISTS
                FOR (n:NODE)
                ON (n.embedding)
                OPTIONS {indexConfig: {
                    `vector.dimensions`: 384,
                    `vector.similarity_function`: 'cosine'
                }}
                """
            )

    async def run_inference(self, repo_id: str):
        docstrings = await self.generate_docstrings(repo_id)
        logger.info(
            f"DEBUGNEO4J: After generate docstrings, Repo ID: {repo_id}, Docstrings: {len(docstrings)}"
        )
        self.log_graph_stats(repo_id)
        self.create_vector_index()

    def query_vector_index(
        self,
        project_id: str,
        query: str,
        node_ids: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> List[Dict]:
        embedding = self.generate_embedding(query)

        with self.driver.session() as session:
            if node_ids:
                # Fetch context node IDs
                result_neighbors = session.run(
                    """
                    MATCH (n:NODE)
                    WHERE n.repoId = $project_id AND n.node_id IN $node_ids
                    CALL {
                        WITH n
                        MATCH (n)-[*1..4]-(neighbor:NODE)
                        RETURN COLLECT(DISTINCT neighbor.node_id) AS neighbor_ids
                    }
                    RETURN COLLECT(DISTINCT n.node_id) + REDUCE(acc = [], neighbor_ids IN COLLECT(neighbor_ids) | acc + neighbor_ids) AS context_node_ids
                    """,
                    project_id=project_id,
                    node_ids=node_ids,
                )
                context_node_ids = result_neighbors.single()["context_node_ids"]

                # Use vector index and filter by context_node_ids
                result = session.run(
                    """
                    CALL db.index.vector.queryNodes('docstring_embedding', $initial_k, $embedding)
                    YIELD node, score
                    WHERE node.repoId = $project_id AND node.node_id IN $context_node_ids
                    RETURN node.node_id AS node_id,
                        node.docstring AS docstring,
                        node.file_path AS file_path,
                        node.start_line AS start_line,
                        node.end_line AS end_line,
                        score AS similarity
                    ORDER BY similarity DESC
                    LIMIT $top_k
                    """,
                    project_id=project_id,
                    embedding=embedding,
                    context_node_ids=context_node_ids,
                    initial_k=top_k * 10,  # Adjust as needed
                    top_k=top_k,
                )
            else:
                result = session.run(
                    """
                    CALL db.index.vector.queryNodes('docstring_embedding', $top_k, $embedding)
                    YIELD node, score
                    WHERE node.repoId = $project_id
                    RETURN node.node_id AS node_id,
                        node.docstring AS docstring,
                        node.file_path AS file_path,
                        node.start_line AS start_line,
                        node.end_line AS end_line,
                        score AS similarity
                    """,
                    project_id=project_id,
                    embedding=embedding,
                    top_k=top_k,
                )

            # Ensure all fields are included in the final output
            return [dict(record) for record in result]









# Setup a dummy SQLAlchemy session (using SQLite in-memory for simplicity)
engine = create_engine("sqlite:///:memory:")
SessionLocal = sessionmaker(bind=engine)
db_session = SessionLocal()

# Neo4j connection
neo4j_uri = NEO4J_URI
neo4j_user = NEO4J_USER
neo4j_password = NEO4J_PASSWORD

# Instantiate service
graph_service = CodeGraphService(neo4j_uri, neo4j_user, neo4j_password, db_session)
InferenceService_service = InferenceService(db_session)
import os
import ast
import tempfile
import shutil
from git import Repo
# Define dummy repo and user info
def clone_repo(git_url):
    temp_dir = tempfile.mkdtemp()
    Repo.clone_from(git_url, temp_dir)
    return temp_dir


# repo_dir =clone_repo("https://github.com/mirzauu/potpie")  # Replace with a real path
project_id = "test_project_1"
# user_id = "user_123"
# # --- Run graph creation ---
# graph_service.create_and_store_graph(repo_dir, project_id, user_id)

# # --- Test node query ---
# from hashlib import md5
# node_id = md5(f"{user_id}:/your/file/path.py".encode("utf-8")).hexdigest()
# node = graph_service.query_graph(
#     f"MATCH (n:NODE {{node_id: '{node_id}', repoId: '{project_id}'}}) RETURN n"
# )
# print(node)

# # Optional cleanup

# graph_service.close()

asyncio.run(InferenceService_service.run_inference(project_id))