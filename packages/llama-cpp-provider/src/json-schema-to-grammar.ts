/**
 * JSON Schema to GBNF Grammar Converter
 *
 * This is a TypeScript port of llama.cpp's json-schema-to-grammar.mjs
 * which was originally ported from json_schema_to_grammar.py
 *
 * Converts JSON Schema to GBNF (GGML BNF) grammar for constraining LLM outputs.
 */

import type { JSONSchema7 } from "@ai-sdk/provider";

/* eslint-disable @typescript-eslint/no-explicit-any */

export interface SchemaConverterOptions {
  /** Property order for object properties */
  propOrder?: Record<string, number>;
  /** Whether to allow fetching remote schemas (not supported in this port) */
  allowFetch?: boolean;
  /** Whether dot in regex patterns matches newlines */
  dotall?: boolean;
}

const SPACE_RULE = '| " " | "\\n"{1,2} [ \\t]{0,20}';

interface RepetitionOptions {
  separatorRule?: string;
  itemRuleIsLiteral?: boolean;
}

function buildRepetition(
  itemRule: string,
  minItems: number,
  maxItems: number | undefined,
  opts: RepetitionOptions = {}
): string {
  if (maxItems === 0) {
    return "";
  }
  if (minItems === 0 && maxItems === 1) {
    return `${itemRule}?`;
  }

  const separatorRule = opts.separatorRule ?? "";

  if (separatorRule === "") {
    if (minItems === 1 && maxItems === undefined) {
      return `${itemRule}+`;
    } else if (minItems === 0 && maxItems === undefined) {
      return `${itemRule}*`;
    } else {
      return `${itemRule}{${minItems},${maxItems !== undefined ? maxItems : ""}}`;
    }
  }

  const result =
    itemRule +
    " " +
    buildRepetition(
      `(${separatorRule} ${itemRule})`,
      minItems > 0 ? minItems - 1 : 0,
      maxItems !== undefined ? maxItems - 1 : undefined
    );
  return minItems === 0 ? `(${result})?` : result;
}

function generateMinMaxInt(
  minValue: number | null,
  maxValue: number | null,
  out: string[],
  decimalsLeft: number = 16,
  topLevel: boolean = true
): void {
  const hasMin = minValue !== null;
  const hasMax = maxValue !== null;

  function digitRange(fromChar: string, toChar: string): void {
    out.push("[");
    if (fromChar === toChar) {
      out.push(fromChar);
    } else {
      out.push(fromChar);
      out.push("-");
      out.push(toChar);
    }
    out.push("]");
  }

  function moreDigits(minDigits: number, maxDigits: number): void {
    out.push("[0-9]");
    if (minDigits === maxDigits && minDigits === 1) {
      return;
    }
    out.push("{");
    out.push(minDigits.toString());
    if (maxDigits !== minDigits) {
      out.push(",");
      if (maxDigits !== Number.MAX_SAFE_INTEGER) {
        out.push(maxDigits.toString());
      }
    }
    out.push("}");
  }

  function uniformRange(fromStr: string, toStr: string): void {
    let i = 0;
    while (i < fromStr.length && fromStr[i] === toStr[i]) {
      i++;
    }
    if (i > 0) {
      out.push('"');
      out.push(fromStr.slice(0, i));
      out.push('"');
    }
    if (i < fromStr.length) {
      if (i > 0) {
        out.push(" ");
      }
      const subLen = fromStr.length - i - 1;
      if (subLen > 0) {
        const fromSub = fromStr.slice(i + 1);
        const toSub = toStr.slice(i + 1);
        const subZeros = "0".repeat(subLen);
        const subNines = "9".repeat(subLen);

        let toReached = false;
        out.push("(");
        if (fromSub === subZeros) {
          digitRange(fromStr[i], String.fromCharCode(toStr.charCodeAt(i) - 1));
          out.push(" ");
          moreDigits(subLen, subLen);
        } else {
          out.push("[");
          out.push(fromStr[i]);
          out.push("] ");
          out.push("(");
          uniformRange(fromSub, subNines);
          out.push(")");
          if (fromStr.charCodeAt(i) < toStr.charCodeAt(i) - 1) {
            out.push(" | ");
            if (toSub === subNines) {
              digitRange(
                String.fromCharCode(fromStr.charCodeAt(i) + 1),
                toStr[i]
              );
              toReached = true;
            } else {
              digitRange(
                String.fromCharCode(fromStr.charCodeAt(i) + 1),
                String.fromCharCode(toStr.charCodeAt(i) - 1)
              );
            }
            out.push(" ");
            moreDigits(subLen, subLen);
          }
        }
        if (!toReached) {
          out.push(" | ");
          digitRange(toStr[i], toStr[i]);
          out.push(" ");
          uniformRange(subZeros, toSub);
        }
        out.push(")");
      } else {
        out.push("[");
        out.push(fromStr[i]);
        out.push("-");
        out.push(toStr[i]);
        out.push("]");
      }
    }
  }

  if (hasMin && hasMax) {
    if (minValue! < 0 && maxValue! < 0) {
      out.push('"-" (');
      generateMinMaxInt(-maxValue!, -minValue!, out, decimalsLeft, true);
      out.push(")");
      return;
    }

    if (minValue! < 0) {
      out.push('"-" (');
      generateMinMaxInt(0, -minValue!, out, decimalsLeft, true);
      out.push(") | ");
      minValue = 0;
    }

    let minS = minValue!.toString();
    const maxS = maxValue!.toString();
    const minDigits = minS.length;
    const maxDigits = maxS.length;

    for (let digits = minDigits; digits < maxDigits; digits++) {
      uniformRange(minS, "9".repeat(digits));
      minS = "1" + "0".repeat(digits);
      out.push(" | ");
    }
    uniformRange(minS, maxS);
    return;
  }

  const lessDecimals = Math.max(decimalsLeft - 1, 1);

  if (hasMin) {
    if (minValue! < 0) {
      out.push('"-" (');
      generateMinMaxInt(null, -minValue!, out, decimalsLeft, false);
      out.push(") | [0] | [1-9] ");
      moreDigits(0, decimalsLeft - 1);
    } else if (minValue === 0) {
      if (topLevel) {
        out.push("[0] | [1-9] ");
        moreDigits(0, lessDecimals);
      } else {
        moreDigits(1, decimalsLeft);
      }
    } else if (minValue! <= 9) {
      const c = minValue!.toString();
      const rangeStart = topLevel ? "1" : "0";
      if (c > rangeStart) {
        digitRange(rangeStart, String.fromCharCode(c.charCodeAt(0) - 1));
        out.push(" ");
        moreDigits(1, lessDecimals);
        out.push(" | ");
      }
      digitRange(c, "9");
      out.push(" ");
      moreDigits(0, lessDecimals);
    } else {
      const minS = minValue!.toString();
      const length = minS.length;
      const c = minS[0];

      if (c > "1") {
        digitRange(
          topLevel ? "1" : "0",
          String.fromCharCode(c.charCodeAt(0) - 1)
        );
        out.push(" ");
        moreDigits(length, lessDecimals);
        out.push(" | ");
      }
      digitRange(c, c);
      out.push(" (");
      generateMinMaxInt(
        parseInt(minS.slice(1)),
        null,
        out,
        lessDecimals,
        false
      );
      out.push(")");
      if (c < "9") {
        out.push(" | ");
        digitRange(String.fromCharCode(c.charCodeAt(0) + 1), "9");
        out.push(" ");
        moreDigits(length - 1, lessDecimals);
      }
    }
    return;
  }

  if (hasMax) {
    if (maxValue! >= 0) {
      if (topLevel) {
        out.push('"-" [1-9] ');
        moreDigits(0, lessDecimals);
        out.push(" | ");
      }
      generateMinMaxInt(0, maxValue, out, decimalsLeft, true);
    } else {
      out.push('"-" (');
      generateMinMaxInt(-maxValue!, null, out, decimalsLeft, false);
      out.push(")");
    }
    return;
  }

  throw new Error("At least one of minValue or maxValue must be set");
}

class BuiltinRule {
  content: string;
  deps: string[];

  constructor(content: string, deps: string[] = []) {
    this.content = content;
    this.deps = deps;
  }
}

const PRIMITIVE_RULES: Record<string, BuiltinRule> = {
  boolean: new BuiltinRule('("true" | "false") space', []),
  "decimal-part": new BuiltinRule("[0-9]{1,16}", []),
  "integral-part": new BuiltinRule("[0] | [1-9] [0-9]{0,15}", []),
  number: new BuiltinRule(
    '("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space',
    ["integral-part", "decimal-part"]
  ),
  integer: new BuiltinRule('("-"? integral-part) space', ["integral-part"]),
  value: new BuiltinRule("object | array | string | number | boolean | null", [
    "object",
    "array",
    "string",
    "number",
    "boolean",
    "null",
  ]),
  object: new BuiltinRule(
    '"{" space ( string ":" space value ("," space string ":" space value)* )? "}" space',
    ["string", "value"]
  ),
  array: new BuiltinRule('"[" space ( value ("," space value)* )? "]" space', [
    "value",
  ]),
  uuid: new BuiltinRule(
    '"\\"" [0-9a-fA-F]{8} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{12} "\\"" space',
    []
  ),
  char: new BuiltinRule(
    `[^"\\\\\\x7F\\x00-\\x1F] | [\\\\] (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})`,
    []
  ),
  string: new BuiltinRule(`"\\"" char* "\\"" space`, ["char"]),
  null: new BuiltinRule('"null" space', []),
};

const STRING_FORMAT_RULES: Record<string, BuiltinRule> = {
  date: new BuiltinRule(
    '[0-9]{4} "-" ( "0" [1-9] | "1" [0-2] ) "-" ( "0" [1-9] | [1-2] [0-9] | "3" [0-1] )',
    []
  ),
  time: new BuiltinRule(
    '([01] [0-9] | "2" [0-3]) ":" [0-5] [0-9] ":" [0-5] [0-9] ( "." [0-9]{3} )? ( "Z" | ( "+" | "-" ) ( [01] [0-9] | "2" [0-3] ) ":" [0-5] [0-9] )',
    []
  ),
  "date-time": new BuiltinRule('date "T" time', ["date", "time"]),
  "date-string": new BuiltinRule('"\\"" date "\\"" space', ["date"]),
  "time-string": new BuiltinRule('"\\"" time "\\"" space', ["time"]),
  "date-time-string": new BuiltinRule('"\\"" date-time "\\"" space', [
    "date-time",
  ]),
};

const RESERVED_NAMES: Record<string, boolean> = {
  root: true,
  ...Object.fromEntries(Object.keys(PRIMITIVE_RULES).map((k) => [k, true])),
  ...Object.fromEntries(Object.keys(STRING_FORMAT_RULES).map((k) => [k, true])),
};

const INVALID_RULE_CHARS_RE = /[^\dA-Za-z-]+/g;
const GRAMMAR_LITERAL_ESCAPE_RE = /[\n\r"\\]/g;
const GRAMMAR_LITERAL_ESCAPES: Record<string, string> = {
  "\r": "\\r",
  "\n": "\\n",
  '"': '\\"',
  "\\": "\\\\",
};

const NON_LITERAL_SET = new Set("|.()[]{}*+?");
const ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS = new Set("^$.[]()|{}*+?");

// Helper function to group elements by a key function
function* groupBy<T, K>(
  iterable: T[],
  keyFn: (item: T) => K
): Generator<[K, T[]]> {
  let lastKey: K | null = null;
  let group: T[] = [];
  for (const element of iterable) {
    const key = keyFn(element);
    if (lastKey !== null && key !== lastKey) {
      yield [lastKey, group];
      group = [];
    }
    group.push(element);
    lastKey = key;
  }
  if (group.length > 0) {
    yield [lastKey!, group];
  }
}

/**
 * Trie node for building string exclusion rules
 */
class TrieNode {
  children: Record<string, TrieNode> = {};
  isEndOfString = false;

  insert(str: string): void {
    let node: TrieNode = this;
    for (const c of str) {
      if (!node.children[c]) {
        node.children[c] = new TrieNode();
      }
      node = node.children[c];
    }
    node.isEndOfString = true;
  }
}

/**
 * SchemaConverter converts JSON Schema to GBNF grammar
 */
export class SchemaConverter {
  private _propOrder: Record<string, number>;
  private _dotall: boolean;
  private _rules: Record<string, string>;
  private _refs: Record<string, JSONSchema7>;
  private _refsBeingResolved: Set<string>;

  constructor(options: SchemaConverterOptions = {}) {
    this._propOrder = options.propOrder || {};
    this._dotall = options.dotall || false;
    this._rules = { space: SPACE_RULE };
    this._refs = {};
    this._refsBeingResolved = new Set();
  }

  private _formatLiteral(literal: string): string {
    const escaped = literal.replace(
      GRAMMAR_LITERAL_ESCAPE_RE,
      (m) => GRAMMAR_LITERAL_ESCAPES[m]
    );
    return `"${escaped}"`;
  }

  private _addRule(name: string, rule: string): string {
    let escName = name.replace(INVALID_RULE_CHARS_RE, "-");
    let key = escName;

    if (escName in this._rules) {
      if (this._rules[escName] === rule) {
        return key;
      }

      let i = 0;
      while (
        `${escName}${i}` in this._rules &&
        this._rules[`${escName}${i}`] !== rule
      ) {
        i += 1;
      }
      key = `${escName}${i}`;
    }

    this._rules[key] = rule;
    return key;
  }

  /**
   * Resolve $ref references in the schema
   */
  resolveRefs(schema: JSONSchema7, url: string = ""): void {
    const visit = (n: any): void => {
      if (Array.isArray(n)) {
        n.forEach(visit);
      } else if (typeof n === "object" && n !== null) {
        const ref = n.$ref;
        if (ref !== undefined && !this._refs[ref]) {
          if (ref.startsWith("https://")) {
            throw new Error(
              "Fetching remote schemas is not supported in this implementation"
            );
          } else if (ref.startsWith("#/")) {
            let target: any = schema;
            const fullRef = `${url}${ref}`;
            n.$ref = fullRef;

            const selectors = ref.split("/").slice(1);
            for (const sel of selectors) {
              if (target && sel in target) {
                target = target[sel];
              } else {
                throw new Error(
                  `Error resolving ref ${ref}: ${sel} not in ${JSON.stringify(target)}`
                );
              }
            }

            this._refs[fullRef] = target;
          } else {
            throw new Error(`Unsupported ref ${ref}`);
          }
        } else {
          Object.values(n).forEach(visit);
        }
      }
    };

    visit(schema);
  }

  private _generateUnionRule(
    name: string | undefined,
    altSchemas: JSONSchema7[]
  ): string {
    return altSchemas
      .map((altSchema, i) =>
        this.visit(altSchema, `${name ?? ""}${name ? "-" : "alternative-"}${i}`)
      )
      .join(" | ");
  }

  private _visitPattern(pattern: string, name: string): string {
    if (!pattern.startsWith("^") || !pattern.endsWith("$")) {
      throw new Error('Pattern must start with "^" and end with "$"');
    }
    pattern = pattern.slice(1, -1);
    const subRuleIds: Record<string, string> = {};

    let i = 0;
    const length = pattern.length;

    const getDot = (): string => {
      let rule: string;
      if (this._dotall) {
        rule = "[\\U00000000-\\U0010FFFF]";
      } else {
        rule = "[^\\x0A\\x0D]";
      }
      return this._addRule("dot", rule);
    };

    const toRule = ([s, isLiteral]: [string, boolean]): string =>
      isLiteral ? '"' + s + '"' : s;

    const transform = (): [string, boolean] => {
      const start = i;
      const seq: [string, boolean][] = [];

      const joinSeq = (): [string, boolean] => {
        const ret: [string, boolean][] = [];
        for (const [isLiteral, g] of groupBy(seq, (x) => x[1])) {
          if (isLiteral) {
            ret.push([[...g].map((x) => x[0]).join(""), true]);
          } else {
            ret.push(...g);
          }
        }
        if (ret.length === 1) {
          return ret[0];
        }
        return [ret.map((x) => toRule(x)).join(" "), false];
      };

      while (i < length) {
        const c = pattern[i];
        if (c === ".") {
          seq.push([getDot(), false]);
          i += 1;
        } else if (c === "(") {
          i += 1;
          if (i < length) {
            if (pattern[i] === "?") {
              throw new Error(
                `Unsupported pattern syntax "${pattern[i]}" at index ${i} of /${pattern}/`
              );
            }
          }
          seq.push([`(${toRule(transform())})`, false]);
        } else if (c === ")") {
          i += 1;
          if (start <= 0 || pattern[start - 1] !== "(") {
            throw new Error(
              `Unbalanced parentheses; start = ${start}, i = ${i}, pattern = ${pattern}`
            );
          }
          return joinSeq();
        } else if (c === "[") {
          let squareBrackets = c;
          i += 1;
          while (i < length && pattern[i] !== "]") {
            if (pattern[i] === "\\") {
              squareBrackets += pattern.slice(i, i + 2);
              i += 2;
            } else {
              squareBrackets += pattern[i];
              i += 1;
            }
          }
          if (i >= length) {
            throw new Error(
              `Unbalanced square brackets; start = ${start}, i = ${i}, pattern = ${pattern}`
            );
          }
          squareBrackets += "]";
          i += 1;
          seq.push([squareBrackets, false]);
        } else if (c === "|") {
          seq.push(["|", false]);
          i += 1;
        } else if (c === "*" || c === "+" || c === "?") {
          seq[seq.length - 1] = [toRule(seq[seq.length - 1]) + c, false];
          i += 1;
        } else if (c === "{") {
          let curlyBrackets = c;
          i += 1;
          while (i < length && pattern[i] !== "}") {
            curlyBrackets += pattern[i];
            i += 1;
          }
          if (i >= length) {
            throw new Error(
              `Unbalanced curly brackets; start = ${start}, i = ${i}, pattern = ${pattern}`
            );
          }
          curlyBrackets += "}";
          i += 1;
          const nums = curlyBrackets
            .slice(1, -1)
            .split(",")
            .map((s) => s.trim());
          let minTimes: number;
          let maxTimes: number | undefined;
          if (nums.length === 1) {
            minTimes = parseInt(nums[0], 10);
            maxTimes = minTimes;
          } else {
            if (nums.length !== 2) {
              throw new Error(`Invalid quantifier ${curlyBrackets}`);
            }
            minTimes = nums[0] ? parseInt(nums[0], 10) : 0;
            maxTimes = nums[1] ? parseInt(nums[1], 10) : undefined;
          }

          let [sub, subIsLiteral] = seq[seq.length - 1];

          if (!subIsLiteral) {
            let id = subRuleIds[sub];
            if (id === undefined) {
              id = this._addRule(
                `${name}-${Object.keys(subRuleIds).length + 1}`,
                sub
              );
              subRuleIds[sub] = id;
            }
            sub = id;
          }

          seq[seq.length - 1] = [
            buildRepetition(
              subIsLiteral ? `"${sub}"` : sub,
              minTimes,
              maxTimes,
              {
                itemRuleIsLiteral: subIsLiteral,
              }
            ),
            false,
          ];
        } else {
          let literal = "";
          while (i < length) {
            if (pattern[i] === "\\" && i < length - 1) {
              const next = pattern[i + 1];
              if (ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS.has(next)) {
                i += 1;
                literal += pattern[i];
                i += 1;
              } else {
                literal += pattern.slice(i, i + 2);
                i += 2;
              }
            } else if (pattern[i] === '"') {
              literal += '\\"';
              i += 1;
            } else if (
              !NON_LITERAL_SET.has(pattern[i]) &&
              (i === length - 1 ||
                literal === "" ||
                pattern[i + 1] === "." ||
                !NON_LITERAL_SET.has(pattern[i + 1]))
            ) {
              literal += pattern[i];
              i += 1;
            } else {
              break;
            }
          }
          if (literal !== "") {
            seq.push([literal, true]);
          }
        }
      }

      return joinSeq();
    };

    return this._addRule(
      name,
      '"\\\"" (' + toRule(transform()) + ') "\\\"" space'
    );
  }

  private _notStrings(strings: string[]): string {
    const trie = new TrieNode();
    for (const s of strings) {
      trie.insert(s);
    }

    const charRuleName = this._addPrimitive("char", PRIMITIVE_RULES["char"]);
    const out: string[] = ['["] ( '];

    const visit = (node: TrieNode): void => {
      const rejects: string[] = [];
      let first = true;
      for (const c of Object.keys(node.children).sort()) {
        const child = node.children[c];
        rejects.push(c);
        if (first) {
          first = false;
        } else {
          out.push(" | ");
        }
        out.push(`[${c}]`);
        if (Object.keys(child.children).length > 0) {
          out.push(" (");
          visit(child);
          out.push(")");
        } else if (child.isEndOfString) {
          out.push(` ${charRuleName}+`);
        }
      }
      if (Object.keys(node.children).length > 0) {
        if (!first) {
          out.push(" | ");
        }
        out.push(`[^"${rejects.join("")}] ${charRuleName}*`);
      }
    };

    visit(trie);

    out.push(` )${trie.isEndOfString ? "" : "?"} ["] space`);
    return out.join("");
  }

  private _resolveRef(ref: string): string {
    const refFragment = ref.split("#").pop()!;
    let refName = "ref" + refFragment.replace(/[^a-zA-Z0-9-]+/g, "-");
    if (!(refName in this._rules) && !this._refsBeingResolved.has(ref)) {
      this._refsBeingResolved.add(ref);
      const resolved = this._refs[ref];
      refName = this.visit(resolved, refName);
      this._refsBeingResolved.delete(ref);
    }
    return refName;
  }

  private _generateConstantRule(value: any): string {
    return this._formatLiteral(JSON.stringify(value));
  }

  visit(schema: JSONSchema7, name: string = ""): string {
    const schemaType = schema.type;
    const schemaFormat = schema.format;
    const ruleName =
      name in RESERVED_NAMES ? name + "-" : name === "" ? "root" : name;

    const ref = schema.$ref;
    if (ref !== undefined) {
      return this._addRule(ruleName, this._resolveRef(ref));
    } else if (schema.oneOf || schema.anyOf) {
      return this._addRule(
        ruleName,
        this._generateUnionRule(name, schema.oneOf || schema.anyOf!)
      );
    } else if (Array.isArray(schemaType)) {
      return this._addRule(
        ruleName,
        this._generateUnionRule(
          name,
          schemaType.map((t) => ({ ...schema, type: t }))
        )
      );
    } else if ("const" in schema) {
      return this._addRule(
        ruleName,
        this._generateConstantRule(schema.const) + " space"
      );
    } else if ("enum" in schema) {
      const rule =
        "(" +
        schema
          .enum!.map((v: any) => this._generateConstantRule(v))
          .join(" | ") +
        ") space";
      return this._addRule(ruleName, rule);
    } else if (
      (schemaType === undefined || schemaType === "object") &&
      ("properties" in schema ||
        ("additionalProperties" in schema &&
          schema.additionalProperties !== true))
    ) {
      const required = new Set<string>(schema.required || []);
      const properties = Object.entries(schema.properties ?? {});
      return this._addRule(
        ruleName,
        this._buildObjectRule(
          properties,
          required,
          name,
          schema.additionalProperties
        )
      );
    } else if (
      (schemaType === undefined ||
        schemaType === "object" ||
        schemaType === "string") &&
      "allOf" in schema
    ) {
      const required = new Set<string>();
      const properties: [string, JSONSchema7][] = [];
      const enumSets: Set<any>[] = [];

      const addComponent = (compSchema: JSONSchema7, isRequired: boolean) => {
        const compRef = compSchema.$ref;
        if (compRef !== undefined) {
          compSchema = this._refs[compRef];
        }

        if ("properties" in compSchema) {
          for (const [propName, propSchema] of Object.entries(
            compSchema.properties!
          )) {
            properties.push([propName, propSchema]);
            if (isRequired) {
              required.add(propName);
            }
          }
        }

        if ("enum" in compSchema) {
          enumSets.push(new Set(compSchema.enum || []));
        }
      };

      for (const t of schema.allOf!) {
        if ("anyOf" in t) {
          for (const tt of t.anyOf!) {
            addComponent(tt, false);
          }
        } else {
          addComponent(t, true);
        }
      }

      if (enumSets.length > 0) {
        const enumIntersection = new Set(
          [...enumSets[0]].filter((v) => enumSets.every((s) => s.has(v)))
        );
        if (enumIntersection.size > 0) {
          const sortedEnums = [...enumIntersection].sort((a, b) =>
            String(a).localeCompare(String(b))
          );
          const rule =
            "(" +
            sortedEnums.map((v) => this._generateConstantRule(v)).join(" | ") +
            ") space";
          return this._addRule(ruleName, rule);
        }
      }
      return this._addRule(
        ruleName,
        this._buildObjectRule(properties, required, name, null)
      );
    } else if (
      (schemaType === undefined || schemaType === "array") &&
      ("items" in schema || "prefixItems" in schema)
    ) {
      const items = schema.items ?? schema.prefixItems;
      if (Array.isArray(items)) {
        return this._addRule(
          ruleName,
          '"[" space ' +
            items
              .map((item, i) =>
                this.visit(item, `${name ?? ""}${name ? "-" : ""}tuple-${i}`)
              )
              .join(' "," space ') +
            ' "]" space'
        );
      } else {
        const itemRuleName = this.visit(
          items!,
          `${name ?? ""}${name ? "-" : ""}item`
        );
        const minItems = schema.minItems || 0;
        const maxItems = schema.maxItems;
        return this._addRule(
          ruleName,
          '"[" space ' +
            buildRepetition(itemRuleName, minItems, maxItems, {
              separatorRule: '"," space',
            }) +
            ' "]" space'
        );
      }
    } else if (
      (schemaType === undefined || schemaType === "string") &&
      "pattern" in schema
    ) {
      return this._visitPattern(schema.pattern!, ruleName);
    } else if (
      (schemaType === undefined || schemaType === "string") &&
      /^uuid[1-5]?$/.test(schema.format || "")
    ) {
      return this._addPrimitive(
        ruleName === "root" ? "root" : schemaFormat!,
        PRIMITIVE_RULES["uuid"]
      );
    } else if (
      (schemaType === undefined || schemaType === "string") &&
      `${schema.format}-string` in STRING_FORMAT_RULES
    ) {
      const primName = `${schema.format}-string`;
      return this._addRule(
        ruleName,
        this._addPrimitive(primName, STRING_FORMAT_RULES[primName])
      );
    } else if (
      schemaType === "string" &&
      ("minLength" in schema || "maxLength" in schema)
    ) {
      const charRuleName = this._addPrimitive("char", PRIMITIVE_RULES["char"]);
      const minLen = schema.minLength || 0;
      const maxLen = schema.maxLength;
      return this._addRule(
        ruleName,
        '"\\\"" ' +
          buildRepetition(charRuleName, minLen, maxLen) +
          ' "\\\"" space'
      );
    } else if (
      schemaType === "integer" &&
      ("minimum" in schema ||
        "exclusiveMinimum" in schema ||
        "maximum" in schema ||
        "exclusiveMaximum" in schema)
    ) {
      let minValue: number | null = null;
      let maxValue: number | null = null;
      if ("minimum" in schema) {
        minValue = schema.minimum!;
      } else if ("exclusiveMinimum" in schema) {
        minValue = schema.exclusiveMinimum! + 1;
      }
      if ("maximum" in schema) {
        maxValue = schema.maximum!;
      } else if ("exclusiveMaximum" in schema) {
        maxValue = schema.exclusiveMaximum! - 1;
      }

      const out: string[] = ["("];
      generateMinMaxInt(minValue, maxValue, out);
      out.push(") space");
      return this._addRule(ruleName, out.join(""));
    } else if (schemaType === "object" || Object.keys(schema).length === 0) {
      return this._addRule(
        ruleName,
        this._addPrimitive("object", PRIMITIVE_RULES["object"])
      );
    } else {
      if (!(schemaType! in PRIMITIVE_RULES)) {
        throw new Error(`Unrecognized schema: ${JSON.stringify(schema)}`);
      }
      return this._addPrimitive(
        ruleName === "root" ? "root" : schemaType!,
        PRIMITIVE_RULES[schemaType!]
      );
    }
  }

  private _addPrimitive(name: string, rule: BuiltinRule): string {
    const n = this._addRule(name, rule.content);
    for (const dep of rule.deps) {
      const depRule = PRIMITIVE_RULES[dep] || STRING_FORMAT_RULES[dep];
      if (!depRule) {
        throw new Error(`Rule ${dep} not known`);
      }
      if (!(dep in this._rules)) {
        this._addPrimitive(dep, depRule);
      }
    }
    return n;
  }

  private _buildObjectRule(
    properties: [string, JSONSchema7][],
    required: Set<string>,
    name: string | undefined,
    additionalProperties: boolean | JSONSchema7 | undefined | null
  ): string {
    const propOrder = this._propOrder;
    const sortedProps = properties
      .map(([k]) => k)
      .sort((a, b) => {
        const orderA = propOrder[a] ?? Infinity;
        const orderB = propOrder[b] ?? Infinity;
        return (
          orderA - orderB ||
          properties.findIndex(([k]) => k === a) -
            properties.findIndex(([k]) => k === b)
        );
      });

    const propKvRuleNames: Record<string, string> = {};
    for (const [propName, propSchema] of properties) {
      const propRuleName = this.visit(
        propSchema,
        `${name ?? ""}${name ? "-" : ""}${propName}`
      );
      propKvRuleNames[propName] = this._addRule(
        `${name ?? ""}${name ? "-" : ""}${propName}-kv`,
        `${this._formatLiteral(JSON.stringify(propName))} space ":" space ${propRuleName}`
      );
    }
    const requiredProps = sortedProps.filter((k) => required.has(k));
    const optionalProps = sortedProps.filter((k) => !required.has(k));

    if (additionalProperties) {
      const subName = `${name ?? ""}${name ? "-" : ""}additional`;
      const valueRule =
        additionalProperties !== null &&
        typeof additionalProperties === "object"
          ? this.visit(additionalProperties, `${subName}-value`)
          : this._addPrimitive("value", PRIMITIVE_RULES["value"]);

      const keyRule =
        sortedProps.length === 0
          ? this._addPrimitive("string", PRIMITIVE_RULES["string"])
          : this._addRule(`${subName}-k`, this._notStrings(sortedProps));

      propKvRuleNames["*"] = this._addRule(
        `${subName}-kv`,
        `${keyRule} ":" space ${valueRule}`
      );
      optionalProps.push("*");
    }

    let rule = '"{" space ';
    rule += requiredProps.map((k) => propKvRuleNames[k]).join(' "," space ');

    if (optionalProps.length > 0) {
      rule += " (";
      if (requiredProps.length > 0) {
        rule += ' "," space ( ';
      }

      const getRecursiveRefs = (
        ks: string[],
        firstIsOptional: boolean
      ): string => {
        const [k, ...rest] = ks;
        const kvRuleName = propKvRuleNames[k];
        let res: string;
        const commaRef = `( "," space ${kvRuleName} )`;
        if (firstIsOptional) {
          res = commaRef + (k === "*" ? "*" : "?");
        } else {
          res = kvRuleName + (k === "*" ? " " + commaRef + "*" : "");
        }
        if (rest.length > 0) {
          res +=
            " " +
            this._addRule(
              `${name ?? ""}${name ? "-" : ""}${k}-rest`,
              getRecursiveRefs(rest, true)
            );
        }
        return res;
      };

      rule += optionalProps
        .map((_, i) => getRecursiveRefs(optionalProps.slice(i), false))
        .join(" | ");
      if (requiredProps.length > 0) {
        rule += " )";
      }
      rule += " )?";
    }

    rule += ' "}" space';

    return rule;
  }

  /**
   * Format the grammar rules as a GBNF grammar string
   */
  formatGrammar(): string {
    let grammar = "";
    for (const [name, rule] of Object.entries(this._rules).sort(([a], [b]) =>
      a.localeCompare(b)
    )) {
      grammar += `${name} ::= ${rule}\n`;
    }
    return grammar;
  }
}

/**
 * Convert a JSON Schema to a GBNF grammar string
 */
export function convertJsonSchemaToGrammar(
  schema: JSONSchema7,
  options: SchemaConverterOptions = {}
): string {
  const converter = new SchemaConverter(options);
  converter.resolveRefs(schema);
  converter.visit(schema, "");
  return converter.formatGrammar();
}
