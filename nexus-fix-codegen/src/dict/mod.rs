mod parse;

pub use parse::{ParseError, parse};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldType {
    Data,
    Length,
    NumInGroup,
    Int,
    SeqNum,
    Bool,
    String,
    Char,
    Float,
    Timestamp,
    Date,
    Time,
    MonthYear,
    DayOfMonth,
    TzTime,
    TzTimestamp,
    Tenor,
    MultiChar,
    MultiString,
}

impl FieldType {
    fn from_str(s: &str) -> Self {
        match s {
            "DATA" | "XMLDATA" => Self::Data,
            "LENGTH" => Self::Length,
            "NUMINGROUP" => Self::NumInGroup,
            "INT" => Self::Int,
            "SEQNUM" => Self::SeqNum,
            "BOOLEAN" => Self::Bool,
            "CHAR" => Self::Char,
            "FLOAT" | "PRICE" | "QTY" | "AMT" | "PERCENTAGE" | "PRICEOFFSET" => Self::Float,
            "UTCTIMESTAMP" => Self::Timestamp,
            "UTCDATEONLY" | "LOCALMKTDATE" => Self::Date,
            "UTCTIMEONLY" => Self::Time,
            "MONTHYEAR" => Self::MonthYear,
            "DAYOFMONTH" => Self::DayOfMonth,
            "TZTIMEONLY" => Self::TzTime,
            "TZTIMESTAMP" => Self::TzTimestamp,
            "TENOR" => Self::Tenor,
            "MULTIPLECHARVALUE" => Self::MultiChar,
            // MULTIPLEVALUESTRING is FIX 4.2's name for what 5.0 calls
            // MULTIPLESTRINGVALUE — space-delimited strings, not single chars.
            "MULTIPLESTRINGVALUE" | "MULTIPLEVALUESTRING" => Self::MultiString,
            _ => Self::String,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EnumValue {
    pub value: String,
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct FieldDef {
    pub number: u32,
    pub name: String,
    pub ftype: FieldType,
    pub values: Vec<EnumValue>,
}

impl FieldDef {
    pub fn is_enum(&self) -> bool {
        !self.values.is_empty()
    }

    pub fn single_char(&self) -> bool {
        self.values.iter().all(|v| v.value.len() == 1)
    }
}

#[derive(Debug, Clone)]
pub enum Member {
    Field { name: String, required: bool },
    Component { name: String },
    Group(GroupDef),
}

#[derive(Debug, Clone)]
pub struct GroupDef {
    pub name: String,
    pub required: bool,
    pub members: Vec<Member>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MsgCat {
    Admin,
    App,
}

#[derive(Debug, Clone)]
pub struct MessageDef {
    pub name: String,
    pub msgtype: String,
    pub msgcat: MsgCat,
    pub members: Vec<Member>,
}

#[derive(Debug, Clone)]
pub struct Dictionary {
    pub major: String,
    pub minor: String,
    pub header: Vec<Member>,
    #[allow(dead_code)]
    pub trailer: Vec<Member>,
    pub fields: Vec<FieldDef>,
    pub components: Vec<(String, Vec<Member>)>,
    pub messages: Vec<MessageDef>,
}

impl Dictionary {
    pub fn field(&self, name: &str) -> Option<&FieldDef> {
        self.fields.iter().find(|f| f.name == name)
    }

    pub fn component(&self, name: &str) -> Option<&[Member]> {
        self.components
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, m)| m.as_slice())
    }
}
