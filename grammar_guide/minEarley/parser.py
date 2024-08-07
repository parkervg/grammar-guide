import re
from typing import List, Any, Collection
from dataclasses import dataclass

import exrex
from lark.lexer import TerminalDef, PatternRE, PatternStr, Pattern
from lark.load_grammar import load_grammar

from .earley import Parser
from .earley_exceptions import UnexpectedCharacters, UnexpectedEOF


@dataclass
class Option:
    start: List[str]
    keep_all_tokens: bool = True
    debug: bool = False

    def __post_init__(self):
        if isinstance(self.start, str):
            self.start = [self.start]


@dataclass
class LexerConf:
    terminals: Collection[TerminalDef]
    ignore: Collection[str]
    use_bytes: bool = False
    re_module = re
    g_regex_flags: int = 0

    def __post_init__(self):
        self.terminals_by_name = {t.name: t for t in self.terminals}
        assert len(self.terminals) == len(self.terminals_by_name)


@dataclass
class ParserConf:
    rules: List[Any]
    start: List[str]


class EarleyParser:
    """
    Frontend for Earley parser
    """

    def __init__(self, grammar: str, **options) -> None:
        self.option = Option(**options)

        self.grammar, _ = load_grammar(
            grammar,
            source="<string>",
            import_paths=[],
            global_keep_all_tokens=self.option.keep_all_tokens,
        )
        self.terminals, self.rules, self.ignore_tokens = self.grammar.compile(
            self.option.start, terminals_to_keep=set()
        )

        self.lexer_conf = LexerConf(self.terminals, self.ignore_tokens)
        self.parser_conf = ParserConf(self.rules, self.option.start)
        self.parser = self.build_parser()

    def build_parser(self):
        parser = Parser(self.lexer_conf, self.parser_conf)
        return parser

    @classmethod
    def open(cls, grammar_filename: str, **options):
        with open(grammar_filename, encoding="utf-8") as f:
            return cls(f.read(), **options)

    def parse(self, text: str, start=None):
        if start is None:
            assert (
                len(self.option.start) == 1
            ), "multiple start symbol, please specify one"
            start = self.option.start[0]
        else:
            assert start in self.option.start

        return self.parser.parse(text, start)

    def handle_error(
        self,
        e,
        candidate_limit: int,
    ):
        def regex_to_candidates(regex):
            candidates = set()
            if exrex.count(regex) > 32:
                # if candidate_overflow_strategy == 'ignore':
                #     logger.warning(
                #         f"regex {regex} has too many candidates. ignoring this pattern"
                #     )
                return regex
            # TODO: why is exrex limit not working here?
            # I think 'limit' matches the unique re patterns that are grabbed,
            # Not actual string generations
            # count = 0
            for candidate in exrex.generate(regex, limit=candidate_limit):
                candidates.add(candidate)
                # count += 1
                # if candidate_limit <= count:
                #     break
            return candidates

        def pattern_to_candidates(pattern: Pattern):
            candidates = set()
            has_re = False
            if isinstance(pattern, PatternStr):
                candidates.add(pattern.value)
            elif isinstance(pattern, PatternRE):
                # candidates.update([pattern.value])
                has_re = True
                candidates.update(regex_to_candidates(pattern.value))
            return candidates, has_re

        has_re = False
        if isinstance(e, UnexpectedCharacters):
            candidate_terminals = set()
            for terminal_name in e.allowed:
                terminal_def = self.lexer_conf.terminals_by_name[terminal_name]
                new_candidates, _has_re = pattern_to_candidates(terminal_def.pattern)
                has_re = _has_re or has_re
                candidate_terminals.update(new_candidates)
            prefix = e.parsed_prefix

            # TODO: handle case where no candidate is found
            if len(candidate_terminals) == 0:
                candidate_terminals = {""}

        elif isinstance(e, UnexpectedEOF):
            candidate_terminals = set()
            for terminal_name in e.expected:
                terminal_def = self.lexer_conf.terminals_by_name[terminal_name]
                new_candidates, _has_re = pattern_to_candidates(terminal_def.pattern)
                has_re = _has_re or has_re
                candidate_terminals.update(new_candidates)

            assert len(candidate_terminals) > 0
            prefix = e.text
        else:
            raise e
        return (prefix, list(candidate_terminals), has_re, e.pos_in_stream)
